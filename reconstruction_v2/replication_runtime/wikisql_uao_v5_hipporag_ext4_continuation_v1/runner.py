"""Run the missing WikiSQL v5 HippoRAG arm from an ext4 mutable root.

Two earlier infrastructure invocations are preserved as zero-effect evidence:

* the first launched four workers which blocked before any index or action;
* the second blocked while fsyncing its claim, before creating ``work``.

This controller binds both records, reuses the exact frozen Agent and RAW
actions, runs only the qualified patched official HippoRAG worker, opens
labels after the three-arm barrier, and invokes the original offline scorer.
It is an append-only continuation in the same v5 lineage, not v6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence

try:
    import base_runner as core
except ModuleNotFoundError:  # Local tests import the source package directly.
    from replication_runtime.wikisql_uao_v5_hipporag_recovery_v1 import (
        runner as core,
    )


VERSION = "wikisql_uao_v5_hipporag_ext4_continuation_v1"
CONFIG_SCHEMA = f"{VERSION}_config_v1"
CONTINUATION_SCHEMA = f"{VERSION}_receipt_v1"
HIPPO_RECEIPT_SCHEMA = f"{VERSION}_hipporag_aggregate_receipt_v1"
BARRIER_SCHEMA = f"{VERSION}_three_arm_barrier_v1"
TERMINAL_SCHEMA = f"{VERSION}_safe_terminal_v1"
FAILURE_SCHEMA = f"{VERSION}_safe_failure_terminal_v1"
STUDY_ID = core.STUDY_ID
ROOT = Path(
    "/var/tmp/wikisql_uao_p4_v5_ext4_continuation_20260730"
)
FIRST_INVOCATION = "a1a4ba82c5974d6b93d58bfc78c92225"
SECOND_INVOCATION = "0831b54786034a76afb414b153368372"
_INVOCATION = re.compile(r"[0-9a-f]{32}\Z")
_CONFIG_KEYS = frozenset(
    {
        "bindings",
        "continuation",
        "lane_assignments",
        "lineage",
        "recovery_root",
        "schema",
        "self_sha256",
        "shard_count",
        "study_id",
        "unit_name",
        "user_authorization",
    }
)

RecoveryError = core.RecoveryError
RecoveryConfig = core.RecoveryConfig
FileBinding = core.FileBinding
TreeBinding = core.TreeBinding


def _activate_core_schemas() -> None:
    """Bind schemas used by reused helper functions in this process."""
    core.VERSION = VERSION
    core.CONFIG_SCHEMA = CONFIG_SCHEMA
    core.CONTINUATION_SCHEMA = CONTINUATION_SCHEMA
    core.HIPPO_RECEIPT_SCHEMA = HIPPO_RECEIPT_SCHEMA
    core.BARRIER_SCHEMA = BARRIER_SCHEMA
    core.TERMINAL_SCHEMA = TERMINAL_SCHEMA
    core.FAILURE_SCHEMA = FAILURE_SCHEMA


def _mount_fstype(path: Path) -> str:
    resolved = path.resolve(strict=True)
    best: tuple[int, str] | None = None
    try:
        rows = Path("/proc/self/mountinfo").read_text(
            encoding="utf-8"
        ).splitlines()
    except OSError as exc:
        raise RecoveryError("mount table is unavailable") from exc
    for row in rows:
        left, separator, right = row.partition(" - ")
        if not separator:
            continue
        fields = left.split()
        trailing = right.split()
        if len(fields) < 5 or not trailing:
            continue
        mount = Path(fields[4].replace("\\040", " "))
        try:
            resolved.relative_to(mount)
        except ValueError:
            continue
        candidate = (len(mount.parts), trailing[0])
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise RecoveryError("mutable-root mount cannot be identified")
    return best[1]


def load_config(path: Path) -> RecoveryConfig:
    value = core._read_json(path, field="ext4 continuation config")
    if set(value) != _CONFIG_KEYS:
        raise RecoveryError("ext4 continuation config shape drifted")
    supplied = core._hex64(value["self_sha256"], "config self")
    body = {key: child for key, child in value.items() if key != "self_sha256"}
    if core.semantic_sha256(body) != supplied:
        raise RecoveryError("ext4 continuation config self hash drifted")
    root = core._absolute(value["recovery_root"], "recovery root")
    if root != ROOT or path != root / "control/continuation_config.json":
        raise RecoveryError("ext4 continuation root or config path drifted")
    if _mount_fstype(root) != "ext4":
        raise RecoveryError("mutable continuation root is not on ext4")
    shard_count = value["shard_count"]
    lanes = value["lane_assignments"]
    if (
        value["schema"] != CONFIG_SCHEMA
        or value["study_id"] != STUDY_ID
        or value["lineage"] != "formal_v5_repair_r1"
        or not isinstance(value["unit_name"], str)
        or isinstance(shard_count, bool)
        or shard_count != 4
        or not isinstance(lanes, list)
        or tuple(lanes) != ("0", "0", "1", "1")
        or value["user_authorization"]
        != {
            "authorized": True,
            "entire_mutable_root_ext4": True,
            "no_v6": True,
            "reuse_agent_and_raw": True,
            "run_only_missing_patched_hipporag_then_offline_score": True,
            "self_recover_after_user_reboot": True,
        }
    ):
        raise RecoveryError("ext4 continuation identity drifted")
    bindings = value["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != {"files", "trees"}:
        raise RecoveryError("binding registry drifted")
    required_files = {
        "agent_action",
        "agent_policy",
        "agent_receipt",
        "base_runner",
        "bound_worker",
        "ext4_overlay_receipt",
        "original_config",
        "original_model_alias_receipt",
        "original_terminal",
        "original_unit",
        "post_reboot_zero_effect_evidence",
        "preindex_dstate_evidence",
        "prior_attempt",
        "prior_continuation_claim",
        "prior_continuation_config",
        "prior_continuation_runner",
        "prior_continuation_unit",
        "prior_intent",
        "prior_recovery_config",
        "prior_recovery_unit",
        "raw_action",
        "recovery_runner",
        "recovery_unit",
        "source_labels",
        "source_view",
    }
    required_trees = {"ext4_overlay", "original_code", "patched_import"}
    raw_files = bindings["files"]
    raw_trees = bindings["trees"]
    if (
        not isinstance(raw_files, dict)
        or set(raw_files) != required_files
        or not isinstance(raw_trees, dict)
        or set(raw_trees) != required_trees
    ):
        raise RecoveryError("binding names drifted")
    files = {
        name: FileBinding.parse(raw_files[name], name)
        for name in sorted(required_files)
    }
    trees = {
        name: TreeBinding.parse(raw_trees[name], name)
        for name in sorted(required_trees)
    }
    if files["recovery_runner"].path.resolve() != Path(__file__).resolve():
        raise RecoveryError("executed ext4 continuation runner is not frozen")
    if files["base_runner"].path.resolve() != Path(core.__file__).resolve():
        raise RecoveryError("executed base runner is not frozen")
    continuation = value["continuation"]
    if continuation != {
        "authorized_after_two_zero_effect_infrastructure_stops": True,
        "effect_retry_or_resample_count": 0,
        "entire_mutable_root_moved_to_ext4": True,
        "prior_infrastructure_invocations": [
            FIRST_INVOCATION,
            SECOND_INVOCATION,
        ],
        "prior_total_shard_process_launch_count": 4,
        "reuse_existing_attempt_intent_agent_and_raw": True,
        "runtime_substitution_only": (
            "exact_dependencies_and_all_mutable_outputs_on_ext4"
        ),
    }:
        raise RecoveryError("ext4 continuation contract drifted")
    return RecoveryConfig(
        path=path,
        recovery_root=root,
        unit_name=value["unit_name"],
        shard_count=shard_count,
        lane_assignments=tuple(lanes),
        files=files,
        trees=trees,
        self_sha256=supplied,
        continuation=continuation,
    )


def _verify_second_zero_effect(
    config: RecoveryConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    claim = core._verify_self_hashed_json(
        config.files["prior_continuation_claim"],
        field="prior continuation claim",
    )
    evidence = core._verify_self_hashed_json(
        config.files["post_reboot_zero_effect_evidence"],
        field="post-reboot zero-effect evidence",
    )
    old_root = Path(
        "/home/erzhu419/wikisql_uao_p4_20260729/"
        "formal_v5_repair_r1_hipporag_recovery"
    )
    if (
        claim.get("schema")
        != "wikisql_uao_v5_hipporag_continuation_v1_receipt_v1"
        or claim.get("status") != "continuation_claimed_once"
        or claim.get("config_file_sha256")
        != config.files["prior_continuation_config"].sha256
        or claim.get("invocation_id_sha256")
        != hashlib.sha256(SECOND_INVOCATION.encode("ascii")).hexdigest()
        or claim.get("effect_retry_or_resample_count") != 0
        or claim.get("A_hold_label_open_count_before_three_arm_barrier") != 0
        or claim.get("Agent_and_RAW_rerun_count") != 0
        or evidence.get("schema")
        != (
            "wikisql_uao_v5_hipporag_continuation_v1_"
            "post_reboot_zero_effect_evidence_v1"
        )
        or evidence.get("status")
        != "PRESERVED_PREFLIGHT_FSYNC_INFRASTRUCTURE_INVALID_ZERO_EFFECT"
        or evidence.get("service_invocation_id") != SECOND_INVOCATION
        or evidence.get("continuation_claim_file_sha256")
        != config.files["prior_continuation_claim"].sha256
        or evidence.get("continuation_claim_self_sha256")
        != claim["self_sha256"]
        or evidence.get("effect_measurement_consumed") is not False
        or evidence.get("effect_retry_or_resample_count") != 0
        or evidence.get("shard_launch_count") != 0
        or evidence.get("completed_HippoRAG_shard_count") != 0
        or evidence.get("index_directory_count") != 0
        or evidence.get("action_file_count") != 0
        or evidence.get("action_barrier_count") != 0
        or evidence.get("source_label_open_count") != 0
        or evidence.get("label_projection_count") != 0
        or evidence.get("scorer_launch_count") != 0
        or evidence.get("Agent_and_RAW_rerun_count") != 0
        or evidence.get("API_or_online_evaluation_count") != 0
        or (old_root / "work").exists()
        or (old_root / "control/outer_terminal.safe.json").exists()
    ):
        raise RecoveryError("second zero-effect evidence drifted")
    return claim, evidence


def _verify_static_bindings(
    config: RecoveryConfig,
    formal: Any,
) -> tuple[Any, tuple[Any, ...], tuple[dict[str, object], ...]]:
    original_config, first_evidence = core._verify_static_bindings(
        config,
        formal,
    )
    second_evidence = _verify_second_zero_effect(config)
    return original_config, first_evidence, second_evidence


def run(config_path: Path) -> Mapping[str, object]:
    _activate_core_schemas()
    config = load_config(config_path)
    root = config.recovery_root
    control = root / "control"
    terminal_path = control / "outer_terminal.safe.json"
    state = core.State()
    if terminal_path.exists() or terminal_path.is_symlink():
        raise RecoveryError("ext4 continuation terminal already exists")
    try:
        action_runtime, scorer, formal, _contract, alias_runtime = (
            core._bootstrap_original(config)
        )
        state.stage = "attest_unique_ext4_user_service"
        service = core._service_attestation(config)
        state.stage = "verify_all_frozen_inputs_and_two_zero_effect_stops"
        original_config, first_evidence, second_evidence = (
            _verify_static_bindings(config, formal)
        )
        first_attempt, prior_intent, preindex = first_evidence
        prior_claim, post_reboot = second_evidence
        continuation = core.self_hashed(
            {
                "API_or_online_evaluation_count": 0,
                "Agent_and_RAW_rerun_count": 0,
                "A_hold_label_open_count_before_three_arm_barrier": 0,
                "config_file_sha256": core.file_sha256(config.path),
                "config_self_sha256": config.self_sha256,
                "effect_retry_or_resample_count": 0,
                "entire_mutable_root_fstype": "ext4",
                "entire_mutable_root_path": str(root),
                "infrastructure_continuation_count": 2,
                "invocation_id_sha256": hashlib.sha256(
                    service["InvocationID"].encode("ascii")
                ).hexdigest(),
                "nrestarts": 0,
                "post_reboot_zero_effect_file_sha256": config.files[
                    "post_reboot_zero_effect_evidence"
                ].sha256,
                "post_reboot_zero_effect_self_sha256": post_reboot[
                    "self_sha256"
                ],
                "preindex_evidence_file_sha256": config.files[
                    "preindex_dstate_evidence"
                ].sha256,
                "preindex_evidence_self_sha256": preindex["self_sha256"],
                "prior_continuation_claim_file_sha256": config.files[
                    "prior_continuation_claim"
                ].sha256,
                "prior_continuation_claim_self_sha256": prior_claim[
                    "self_sha256"
                ],
                "prior_infrastructure_invocations": [
                    FIRST_INVOCATION,
                    SECOND_INVOCATION,
                ],
                "prior_total_shard_process_launch_count": 4,
                "protocol_exception": (
                    "user_authorized_same_v5_two_zero_effect_"
                    "infrastructure_continuations"
                ),
                "runtime_substitution_only": (
                    "exact_dependencies_and_all_mutable_outputs_on_ext4"
                ),
                "schema": CONTINUATION_SCHEMA,
                "status": "ext4_continuation_claimed_once",
                "study_id": STUDY_ID,
            }
        )
        continuation_file = core._write_once(
            control / "continuation.safe.json",
            continuation,
        )
        state.stage = "prepare_four_label_free_shards"
        work = root / "work"
        core._mkdir(work)
        core._mkdir(work / "hippo")
        full_view, full_items, shard_views = core._make_shards(
            config=config,
            action_runtime=action_runtime,
            work=work,
        )
        state.stage = "launch_four_patched_HippoRAG_shards"
        state.shard_launch_count = config.shard_count
        shard_rows, shard_receipts = core._launch_shards(
            config=config,
            action_runtime=action_runtime,
            original_config=original_config,
            formal=formal,
            alias_runtime=alias_runtime,
            shard_views=shard_views,
            work=work,
        )
        state.stage = "seal_full_HippoRAG_action_pack"
        hippo, hippo_receipt = core._seal_hippo(
            config=config,
            action_runtime=action_runtime,
            full_view=full_view,
            full_items=full_items,
            shard_rows=shard_rows,
            shard_receipts=shard_receipts,
            work=work,
        )
        state.stage = "validate_and_seal_three_arm_barrier"
        actions = core._validate_existing_arms(
            config=config,
            action_runtime=action_runtime,
            view_sha256=full_view["self_sha256"],
            hippo=hippo,
        )
        barrier = core.self_hashed(
            {
                "A_hold_label_open_count_before_barrier": 0,
                "Agent_and_RAW_reused_byte_exact": True,
                "all_three_actions_durable": True,
                "action_commitments": actions,
                "continuation_file_sha256": continuation_file,
                "continuation_self_sha256": continuation["self_sha256"],
                "hipporag_receipt_self_sha256": hippo_receipt["self_sha256"],
                "intent_file_sha256": config.files["prior_intent"].sha256,
                "intent_self_sha256": prior_intent["self_sha256"],
                "schema": BARRIER_SCHEMA,
                "status": "three_common_action_packs_sealed",
                "study_id": STUDY_ID,
            }
        )
        barrier_path = control / "action_barrier.safe.json"
        barrier_file = core._write_once(barrier_path, barrier)
        state.action_barrier_count = 1
        state.stage = "post_barrier_project_minimal_labels"
        scorer_root = work / "scorer"
        core._mkdir(scorer_root)
        for child in ("home", "tmp"):
            core._mkdir(scorer_root / child)
        label_file = core._project_labels(
            config=config,
            scorer=scorer,
            view_sha256=full_view["self_sha256"],
            output=scorer_root / "A_hold.minimal.labels.json",
        )
        state.label_projection_count = 1
        state.stage = "run_original_offline_scorer_once"
        state.scorer_launch_count = 1
        if (
            core._run_scorer(
                config=config,
                original_config=original_config,
                formal=formal,
                scorer_root=scorer_root,
                barrier=barrier_path,
            )
            != 0
        ):
            raise RecoveryError("original offline scorer failed")
        state.stage = "validate_safe_aggregate_and_write_terminal"
        scorer_paths = formal.FormalPaths.for_root(root)
        scorer_artifacts = formal._verify_scorer_outputs(scorer_paths)
        terminal = core.self_hashed(
            {
                "API_or_online_evaluation_count": 0,
                "Agent_and_RAW_rerun_count": 0,
                "a_hold_label_opened_only_after_action_barrier": True,
                "a_hold_minimal_label_file_sha256": label_file,
                "action_barrier_file_sha256": barrier_file,
                "action_barrier_self_sha256": barrier["self_sha256"],
                "config_self_sha256": config.self_sha256,
                "effect_retry_or_resample_count": 0,
                "infrastructure_continuation_count": 2,
                "mutable_root_fstype": "ext4",
                "nrestarts": 0,
                "original_failed_terminal_preserved": True,
                "original_terminal_file_sha256": config.files[
                    "original_terminal"
                ].sha256,
                "parallel_HippoRAG_shard_count": config.shard_count,
                "patched_source_sha256": core.PATCHED_SOURCE_SHA256,
                "primary_passed": scorer_artifacts.terminal[
                    "primary_passed"
                ],
                "prior_infrastructure_invocation_count": 2,
                "prior_preindex_HippoRAG_shard_launch_count": 4,
                "prior_prefsync_HippoRAG_shard_launch_count": 0,
                "protocol_exception": (
                    "user_authorized_post_terminal_same_v5_missing_arm_"
                    "ext4_continuation"
                ),
                "schema": TERMINAL_SCHEMA,
                "scorer_safe_aggregate_file_sha256": (
                    scorer_artifacts.safe_receipt_file_sha256
                ),
                "scorer_safe_terminal_file_sha256": (
                    scorer_artifacts.terminal_file_sha256
                ),
                "scorer_safe_terminal_self_sha256": (
                    scorer_artifacts.terminal["self_sha256"]
                ),
                "status": "completed_post_terminal_same_v5_ext4_continuation",
                "study_id": STUDY_ID,
                "total_HippoRAG_process_launch_count": 8,
            }
        )
        core._write_once(terminal_path, terminal)
        return terminal
    except BaseException as exc:
        if not terminal_path.exists() and not terminal_path.is_symlink():
            failure = core.self_hashed(
                {
                    "API_or_online_evaluation_count": 0,
                    "Agent_and_RAW_rerun_count": 0,
                    "action_barrier_count": state.action_barrier_count,
                    "effect_retry_or_resample_count": 0,
                    "failure_fingerprint_sha256": hashlib.sha256(
                        f"{type(exc).__name__}:{exc}".encode(
                            "utf-8", errors="replace"
                        )
                    ).hexdigest(),
                    "infrastructure_continuation_count": 2,
                    "label_projection_count": state.label_projection_count,
                    "mutable_root_fstype": "ext4",
                    "prior_total_shard_process_launch_count": 4,
                    "protocol_exception": True,
                    "schema": FAILURE_SCHEMA,
                    "scorer_launch_count": state.scorer_launch_count,
                    "shard_launch_count": state.shard_launch_count,
                    "stage": state.stage,
                    "status": (
                        "failed_post_terminal_ext4_continuation_"
                        "efficacy_unknown"
                    ),
                    "study_id": STUDY_ID,
                    "total_shard_process_launch_count": (
                        4 + state.shard_launch_count
                    ),
                }
            )
            core._write_once(terminal_path, failure)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)
    terminal = run(arguments.config)
    print(
        json.dumps(
            terminal,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
