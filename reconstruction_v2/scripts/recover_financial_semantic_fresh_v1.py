#!/usr/bin/env python3
"""Outcome-blind recovery for the interrupted financial fresh batch.

The original scheduler disappeared after all ten Codex turns had terminated but
before nine post-agent verifiers ran.  This program never invokes an agent or a
model.  It binds the orphaned state before measurement, resumes the frozen
semantic operator once, materializes the frozen tests, runs all remaining
offline verifiers concurrently, and emits an explicitly recovered report.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks.codex_action_budget import (
    audit_codex_action_budget,
)
from assumption_agent.benchmarks.financial_semantic_fresh_runner_v1 import (
    ACTIVE_FRESH_ITEM_ID,
    FRESH_SPLIT_RELATIVE_PATH,
    OFFICIAL_HIPPORAG_STATUS,
    V320_PROTOCOL_RELATIVE_PATH,
    _prepare_runtime_assets_v1,
    _worker_artifact_closure,
    build_fresh_execution_plan_v1,
    load_fresh_split_metadata_v1,
    load_frozen_financial_treatment_v1,
)
from assumption_agent.benchmarks.financial_semantic_integration_v1 import (
    FinancialSemanticSubprocessBackendV1,
    SharedFinancialSemanticPlannerV1,
    _FinancialRunStateV1,
)
from assumption_agent.benchmarks.offline_verifier import (
    offline_verifier_profile_for_family,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    CODEX_ACTION_SUPERVISOR_PATH,
    SkillLearnSubprocessBackend,
    _directory_content_hash,
)
from assumption_agent.benchmarks.train_execution_contract_development_v2 import (
    SKILLLEARN_BENCHMARK_RELATIVE_ROOT,
)
from assumption_agent.events import Event, JsonlEventSink
from assumption_agent.models import stable_hash
from assumption_agent.benchmarks.paper_protocol import PaperProtocol


RECOVERY_VERSION = "financial_semantic_scheduler_loss_recovery_v1"
PREREG_VERSION = "financial_semantic_scheduler_loss_recovery_prereg_v1"
SCRIPT_RELATIVE_PATH = "scripts/recover_financial_semantic_fresh_v1.py"
ORIGINAL_EVENTS = "execution.events.jsonl"
ORIGINAL_PREFLIGHT = "asset_preflight.report.json"
RECOVERY_EVENTS = "recovery.events.jsonl"
RECOVERY_REPORT = "fresh_paired.recovered.report.json"
RECOVERY_FAILURE = "fresh_paired.recovery.failure.json"
SESSION_RECEIPT = "recovery.session.started.json"
SEMANTIC_STAGE = "recovery.semantic.stage.json"
VERIFIER_STAGE = "recovery.verifier.stage.json"
EXPECTED_PHYSICAL = 10
EXPECTED_ORPHANS = 9
EXPECTED_PRECOMPLETED = 1
EXPECTED_VERIFIER_WORKERS = 9

_MODEL_SECRET_NAMES = (
    "OPENAI_API_KEY",
    "GPT5_API_KEY",
    "RUOLI_API_KEY",
    "ASSUMPTION_V2_API_KEY",
)
_SEMANTIC_TEMP_PATHS = (
    "/tmp/assumption_financial_semantic_operator_v1.py",
    "/tmp/assumption_financial_semantic_plan_v1.json",
    "/tmp/assumption_financial_semantic_receipt_v1.json",
)


class RecoveryError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RecoveryError(f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RecoveryError(f"missing JSON receipt: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecoveryError(f"JSON receipt is not an object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any], *, refuse: bool = False) -> None:
    if refuse and (path.exists() or path.is_symlink()):
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run(
    args: Sequence[str],
    *,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(args),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        raise RecoveryError(
            "command failed: " + stable_hash({"argv": list(args), "exit": completed.returncode})
        )
    return completed


def _git(project: Path, *args: str) -> str:
    return _run(["git", "-C", str(project), *args]).stdout.strip()


def _git_bytes(project: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(project), *args],
        capture_output=True,
        check=True,
    )
    return completed.stdout


def _self_hashed(body: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(body)
    result[key] = stable_hash(body)
    return result


def _relative(project: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project.resolve()).as_posix()
    except ValueError as exc:
        raise RecoveryError("recovery path escaped project root") from exc


def _configure_nonsecret_environment(protocol: PaperProtocol) -> None:
    payload = protocol.payload
    execution = payload["execution"]
    os.environ["ASSUMPTION_V2_API_BASE"] = str(payload["provider_endpoint_origin"])
    os.environ["ASSUMPTION_V2_MODEL"] = str(payload["model"])
    os.environ["ASSUMPTION_V2_SKILLLEARN_PROVIDER_MODE"] = str(
        payload["trial_provider_mode"]
    )
    os.environ["ASSUMPTION_V2_API_ALLOWED_IPV4S"] = ",".join(
        str(value) for value in payload["provider_endpoint_ipv4s"]
    )
    os.environ["ASSUMPTION_V2_SKILLLEARN_CACHE_ONLY"] = "1"
    os.environ["ASSUMPTION_V2_TRIAL_NETWORK_BYTE_LIMIT"] = str(
        execution["trial_network_byte_limit"]
    )
    for name in _MODEL_SECRET_NAMES:
        os.environ.pop(name, None)


def _load_context(
    *, project: Path, batch_root: Path, treatment_manifest: Path
) -> dict[str, Any]:
    protocol = PaperProtocol.read(project / V320_PROTOCOL_RELATIVE_PATH)
    benchmark_root = (project / SKILLLEARN_BENCHMARK_RELATIVE_ROOT).resolve(strict=True)
    split = load_fresh_split_metadata_v1(project / FRESH_SPLIT_RELATIVE_PATH)
    treatment = load_frozen_financial_treatment_v1(
        project_root=project,
        benchmark_root=benchmark_root,
        path=treatment_manifest.resolve(strict=True),
        split=split,
    )
    candidate_source = (project / treatment.candidate_skill_source).resolve(strict=True)
    plan = build_fresh_execution_plan_v1(
        split=split,
        treatment=treatment,
        candidate_skill_source=candidate_source,
        agent_id=str(protocol.payload["agent_id"]),
        model=str(protocol.payload["model"]),
        max_steps=int(protocol.payload["max_steps"]),
        codex_agent_execution_policy_hash=protocol.codex_agent_execution_policy.policy_hash,
    )
    return {
        "project": project,
        "batch_root": batch_root,
        "benchmark_root": benchmark_root,
        "protocol": protocol,
        "split": split,
        "treatment": treatment,
        "candidate_source": candidate_source,
        "plan": plan,
        "trials_root": batch_root / "worker_state",
    }


def _skill_config(work: Any) -> str:
    return "assumption-agent-v2-challenger" if work.arm == "semantic" else "no_skill"


def _container_name(work: Any) -> str:
    safe_id = f"{work.family}-{work.item_id}".replace("/", "-").replace("\\", "-")
    return f"evaluation_{safe_id}_{_skill_config(work)}_{work.request.trial_id.lower()}"


def _trial_path(context: Mapping[str, Any], work: Any) -> Path:
    return (
        context["trials_root"]
        / work.work_unit_hash
        / _skill_config(work)
        / work.family
        / work.item_id
        / work.request.trial_id
    )


def _event_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RecoveryError("event ledger contains a non-object")
        rows.append(value)
    return rows


def _container_inspect(name: str) -> dict[str, Any]:
    raw = _run(["docker", "inspect", name]).stdout
    values = json.loads(raw)
    if not isinstance(values, list) or len(values) != 1:
        raise RecoveryError("container inspect cardinality drifted")
    value = values[0]
    state = value["State"]
    if state.get("Status") != "running" or not state.get("Running"):
        raise RecoveryError("orphan container is not running")
    mounts = []
    for mount in value.get("Mounts", []):
        mounts.append(
            {
                "type": mount.get("Type"),
                "source": mount.get("Name") if mount.get("Type") == "volume" else mount.get("Source"),
                "destination": mount.get("Destination"),
                "read_write": bool(mount.get("RW")),
            }
        )
    return {
        "container_id": str(value["Id"]),
        "container_name": str(value["Name"]).removeprefix("/"),
        "image_id": str(value["Image"]),
        "created": str(value["Created"]),
        "network_mode": str(value["HostConfig"]["NetworkMode"]),
        "mounts": sorted(mounts, key=lambda row: str(row["destination"])),
        "labels": dict(sorted((value["Config"].get("Labels") or {}).items())),
        "secret_environment_persisted": False,
    }


def _idle_process_receipt(name: str) -> dict[str, Any]:
    completed = _run(["docker", "top", name, "-eo", "pid,ppid,stat,comm,args"])
    rows = []
    for index, line in enumerate(completed.stdout.splitlines()):
        if index == 0 or not line.strip():
            continue
        parts = line.split(None, 4)
        if len(parts) < 5:
            raise RecoveryError("container process listing is malformed")
        _, _, stat, command, arguments = parts
        if command not in {"sh", "sleep"}:
            raise RecoveryError("non-idle process remains in orphan container")
        lowered = arguments.lower()
        if any(value in lowered for value in ("codex", "python", "pytest", "node")):
            raise RecoveryError("agent or verifier process remains in orphan container")
        rows.append(
            {
                "stat": stat,
                "command": command,
                "arguments_hash": stable_hash({"arguments": arguments}),
            }
        )
    if not rows:
        raise RecoveryError("orphan container has no idle shim")
    return {"process_count": len(rows), "process_shape_hash": stable_hash({"rows": rows})}


def _container_exists(name: str, path: str) -> bool:
    return _run(["docker", "exec", name, "test", "-e", path], check=False).returncode == 0


def _file_rows(root: Path) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RecoveryError("linked file in receipt tree")
        if path.is_file():
            rows.append((path.relative_to(root).as_posix(), _sha256(path)))
        elif not path.is_dir():
            raise RecoveryError("non-regular receipt tree entry")
    return tuple(rows)


def _reconstruct_installed_candidate_receipt(
    context: Mapping[str, Any], container_name: str
) -> dict[str, Any]:
    source = context["candidate_source"]
    source_rows = _file_rows(source)
    if not source_rows:
        raise RecoveryError("candidate source is empty")
    dockerfile = (
        context["benchmark_root"]
        / "tasks"
        / "financial-analysis"
        / ACTIVE_FRESH_ITEM_ID
        / "environment"
        / "Dockerfile"
    )
    backend = SkillLearnSubprocessBackend(context["benchmark_root"])
    runner = backend._load_runner()
    copies = runner._parse_skill_copies(dockerfile)
    if not copies or any(source_pattern != "skills" for source_pattern, _ in copies):
        raise RecoveryError("candidate skill copy grammar drifted")
    destination_rows = []
    with tempfile.TemporaryDirectory(prefix="financial-recovery-install-") as raw_root:
        root = Path(raw_root)
        staged_source = root / "external-source"
        shutil.copytree(source, staged_source)
        normalized = root / "expected"
        if runner._copy_skills_to_dest(staged_source, normalized) is not True:
            raise RecoveryError("candidate normalization failed")
        expected_rows = _file_rows(normalized)
        for index, (source_pattern, destination) in enumerate(copies):
            installed = root / f"installed-{index}"
            installed.mkdir()
            expected_child = PurePosixPath(destination) / "external-source"
            _run(
                [
                    "docker",
                    "cp",
                    f"{container_name}:{expected_child}",
                    str(installed / "external-source"),
                ]
            )
            observed_rows = _file_rows(installed)
            if observed_rows != expected_rows:
                raise RecoveryError("post-agent candidate installation bytes drifted")
            destination_rows.append(
                {
                    "source_pattern": source_pattern,
                    "destination_hash": stable_hash({"path": destination}),
                    "files": [
                        {"path": path, "sha256": sha256}
                        for path, sha256 in observed_rows
                    ],
                }
            )
    return {
        "source_file_hashes": [list(row) for row in source_rows],
        "installed_tree_hash": stable_hash({"destinations": destination_rows}),
        "destination_count": len(destination_rows),
        "post_agent_expected_subtrees_match": True,
        "agent_launch_implies_fail_closed_pre_agent_receipt_passed": True,
        "receipt_reconstructed_after_scheduler_loss": True,
    }


def _prereg_snapshot(context: Mapping[str, Any]) -> dict[str, Any]:
    events_path = context["batch_root"] / ORIGINAL_EVENTS
    preflight_path = context["batch_root"] / ORIGINAL_PREFLIGHT
    events = _event_rows(events_path)
    completed_request_hashes = {
        str(row.get("payload", {}).get("request_hash") or "")
        for row in events
        if row.get("event") == "skilllearn_trial_completed"
    }
    semantic_work = next(
        work for work in context["plan"].physical_work_units if work.arm == "semantic"
    )
    plan_events = [
        row
        for row in events
        if row.get("event") == "financial_semantic_plan_built_before_agent_v1"
        and row.get("payload", {}).get("request_hash") == semantic_work.request.request_hash
    ]
    if len(plan_events) != 1:
        raise RecoveryError("pre-agent semantic plan event coverage drifted")
    semantic_binding = {
        key: plan_events[0]["payload"][key]
        for key in (
            "request_hash",
            "candidate_id",
            "candidate_manifest_hash",
            "plan_hash",
            "extraction_receipt_hash",
            "instruction_sha256",
        )
    }
    rows = []
    precompleted = 0
    orphans = 0
    for work in context["plan"].physical_work_units:
        trial_path = _trial_path(context, work)
        trace_path = trial_path / "agent" / "codex.txt"
        receipt_path = trial_path / "agent" / "codex_action_budget_receipt.json"
        receipt = _read_json(receipt_path)
        audit = audit_codex_action_budget(
            trace_path=trace_path,
            receipt_path=receipt_path,
            supervisor_path=CODEX_ACTION_SUPERVISOR_PATH,
            expected_limit=int(context["protocol"].payload["max_steps"]),
            expected_process_scope=str(
                context["protocol"].codex_agent_execution_policy.action_budget_process_scope
            ),
        )
        if (
            not audit.valid
            or not audit.token_usage_complete
            or not audit.turn_completed_observed
            or not audit.agent_processes_exit_confirmed
            or receipt.get("agent_exit_code") != 0
        ):
            raise RecoveryError("terminal agent receipt is invalid")
        result_path = trial_path / "result.json"
        common = {
            **work.safe_payload(),
            "trial_id": work.request.trial_id,
            "skill_config": _skill_config(work),
            "trace_sha256": _sha256(trace_path),
            "action_budget_file_sha256": _sha256(receipt_path),
            "action_budget_receipt_hash": audit.receipt_hash,
            "observed_steps": audit.observed_steps,
            "token_usage_complete": audit.token_usage_complete,
            "agent_exit_confirmed": audit.agent_processes_exit_confirmed,
        }
        if result_path.is_file():
            precompleted += 1
            if work.request.request_hash not in completed_request_hashes:
                raise RecoveryError("persisted result lacks original completion event")
            common.update(
                {
                    "state": "precompleted",
                    "result_file_sha256": _sha256(result_path),
                    "container_expected": False,
                }
            )
        else:
            orphans += 1
            if work.request.request_hash in completed_request_hashes:
                raise RecoveryError("orphan unexpectedly has original completion event")
            name = _container_name(work)
            inspect = _container_inspect(name)
            if inspect["container_name"] != name:
                raise RecoveryError("container name drifted")
            if _container_exists(name, "/tests"):
                raise RecoveryError("verifier tests were exposed before preregistration")
            if any(_container_exists(name, path) for path in _SEMANTIC_TEMP_PATHS):
                raise RecoveryError("semantic runtime marker existed before preregistration")
            verifier_dir = trial_path / "verifier"
            if any((verifier_dir / name).exists() for name in ("ctrf.json", "reward.txt")):
                raise RecoveryError("verifier outcome existed before preregistration")
            profile = offline_verifier_profile_for_family(work.family)
            if profile is None:
                raise RecoveryError("offline verifier profile disappeared")
            tests = (
                context["benchmark_root"] / "tasks" / work.family / work.item_id / "tests"
            )
            common.update(
                {
                    "state": "orphan_post_agent_pre_verifier",
                    "container": inspect,
                    "idle_process_receipt": _idle_process_receipt(name),
                    "tests_content_hash": _directory_content_hash(tests),
                    "offline_verifier_profile_id": profile.profile_id,
                    "offline_verifier_profile_hash": profile.profile_hash,
                    "offline_verifier_command_hash": stable_hash(
                        {"command": profile.verifier_command}
                    ),
                    "result_file_present": False,
                    "verifier_outcome_present": False,
                }
            )
        rows.append(common)
    if (
        len(rows) != EXPECTED_PHYSICAL
        or precompleted != EXPECTED_PRECOMPLETED
        or orphans != EXPECTED_ORPHANS
        or semantic_work.request.request_hash in completed_request_hashes
    ):
        raise RecoveryError("interrupted batch topology drifted")
    return {
        "original_event_ledger_sha256": _sha256(events_path),
        "original_event_count": len(events),
        "original_preflight_sha256": _sha256(preflight_path),
        "original_preflight_report_hash": _read_json(preflight_path)["report_hash"],
        "precompleted_count": precompleted,
        "orphan_count": orphans,
        "worker_rows": sorted(rows, key=lambda row: row["work_unit_hash"]),
        "worker_row_set_hash": stable_hash(
            {"rows": sorted(rows, key=lambda row: row["work_unit_hash"])}
        ),
        "semantic_pre_agent_binding": semantic_binding,
    }


def preregister(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project_root.resolve(strict=True)
    batch_root = args.batch_root.resolve(strict=True)
    context = _load_context(
        project=project,
        batch_root=batch_root,
        treatment_manifest=args.treatment_manifest,
    )
    script_path = project / SCRIPT_RELATIVE_PATH
    source_commit = _git(project, "rev-parse", "HEAD")
    snapshot = _prereg_snapshot(context)
    body = {
        "manifest_version": PREREG_VERSION,
        "created_at_utc": _utc_now(),
        "recovery_version": RECOVERY_VERSION,
        "recovery_script_relative_path": SCRIPT_RELATIVE_PATH,
        "recovery_script_sha256": _sha256(script_path),
        "recovery_source_commit": source_commit,
        "batch_root": _relative(project, batch_root),
        "treatment_manifest": _relative(project, args.treatment_manifest.resolve(strict=True)),
        "treatment_manifest_hash": context["treatment"].manifest_hash,
        "fresh_split_manifest_hash": context["split"].manifest_hash,
        "execution_plan_hash": context["plan"].plan_hash,
        "snapshot": snapshot,
        "intervention": {
            "scheduler_process_loss": True,
            "agent_stage_complete_for_all_physical_work": True,
            "active_pair_outcomes_accessed_by_recovery_design": False,
            "one_nonactive_precompleted_outcome_was_already_available": True,
            "precompleted_outcome_not_used_for_recovery_decisions": True,
            "semantic_plan_rebuild_must_match_pre_agent_hashes": True,
            "semantic_operator_resume_count": 1,
            "offline_verifier_resume_count": EXPECTED_ORPHANS,
            "offline_verifier_workers": EXPECTED_VERIFIER_WORKERS,
            "model_calls_authorized": 0,
            "model_call_replay_authorized": False,
            "provider_switch_authorized": False,
            "online_evaluation_authorized": False,
            "official_hipporag_execution_authorized": False,
            "outcome_conditioned_branching_authorized": False,
            "new_performance_gate_added": False,
            "container_cleanup_after_artifact_capture": True,
        },
        "claim_boundary": {
            "pristine_runner_completion": False,
            "post_agent_recovery_must_be_disclosed": True,
            "paired_task_utility_interpretation_allowed_if_all_receipts_valid": True,
            "promotion_authorized": False,
            "sealed_test_access_authorized": False,
        },
        "secret_value_persisted": False,
    }
    manifest = _self_hashed(body, "manifest_hash")
    _atomic_json(args.output.resolve(), manifest, refuse=True)
    return manifest


def _verify_prereg(
    context: Mapping[str, Any], manifest_path: Path
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    declared = manifest.pop("manifest_hash", None)
    if declared != stable_hash(manifest):
        raise RecoveryError("recovery preregistration self-hash drifted")
    manifest["manifest_hash"] = declared
    if (
        manifest.get("manifest_version") != PREREG_VERSION
        or manifest.get("recovery_version") != RECOVERY_VERSION
        or manifest.get("execution_plan_hash") != context["plan"].plan_hash
        or manifest.get("treatment_manifest_hash") != context["treatment"].manifest_hash
        or manifest.get("fresh_split_manifest_hash") != context["split"].manifest_hash
    ):
        raise RecoveryError("recovery preregistration identity drifted")
    script_path = context["project"] / SCRIPT_RELATIVE_PATH
    if _sha256(script_path) != manifest.get("recovery_script_sha256"):
        raise RecoveryError("recovery script changed after preregistration")
    source_commit = str(manifest.get("recovery_source_commit") or "")
    committed = _git_bytes(
        context["project"], "show", f"{source_commit}:{SCRIPT_RELATIVE_PATH}"
    )
    if hashlib.sha256(committed).hexdigest() != manifest.get(
        "recovery_script_sha256"
    ):
        raise RecoveryError("preregistered recovery source is not committed")
    snapshot = manifest["snapshot"]
    if (
        _sha256(context["batch_root"] / ORIGINAL_EVENTS)
        != snapshot["original_event_ledger_sha256"]
        or _sha256(context["batch_root"] / ORIGINAL_PREFLIGHT)
        != snapshot["original_preflight_sha256"]
    ):
        raise RecoveryError("original interrupted evidence changed after preregistration")
    return manifest


def _emit(sink: JsonlEventSink, event: str, payload: Mapping[str, Any]) -> None:
    sink.emit(
        Event(
            event=event,
            stage="benchmark.skilllearn.scheduler_loss_recovery_v1",
            trace_id="financial-semantic-fresh-recovery-v1",
            payload=dict(payload),
        )
    )


def _copy_tests_and_verify(
    *, context: Mapping[str, Any], work: Any, sink: JsonlEventSink
) -> dict[str, Any]:
    name = _container_name(work)
    trial_path = _trial_path(context, work)
    verifier_dir = trial_path / "verifier"
    stage_path = verifier_dir / VERIFIER_STAGE
    if stage_path.exists():
        existing = _read_json(stage_path)
        if existing.get("status") == "completed":
            return existing
        raise RecoveryError("ambiguous prior verifier recovery stage")
    profile = offline_verifier_profile_for_family(work.family)
    if profile is None:
        raise RecoveryError("offline verifier profile unavailable")
    tests = context["benchmark_root"] / "tasks" / work.family / work.item_id / "tests"
    source_hash = _directory_content_hash(tests)
    started = _self_hashed(
        {
            "stage_version": RECOVERY_VERSION,
            "status": "started",
            "started_at_utc": _utc_now(),
            "work_unit_hash": work.work_unit_hash,
            "request_hash": work.request.request_hash,
            "container_name_hash": stable_hash({"container_name": name}),
            "tests_content_hash": source_hash,
            "verifier_profile_id": profile.profile_id,
            "verifier_profile_hash": profile.profile_hash,
            "verifier_command_hash": stable_hash({"command": profile.verifier_command}),
            "model_calls": 0,
            "online_judge_calls": 0,
        },
        "stage_hash",
    )
    _atomic_json(stage_path, started, refuse=True)
    _run(["docker", "exec", name, "mkdir", "-p", "/tests"])
    _run(["docker", "cp", f"{tests}/.", f"{name}:/tests"])
    with tempfile.TemporaryDirectory(prefix="financial-recovery-tests-") as raw:
        readback = Path(raw) / "tests"
        readback.mkdir()
        _run(["docker", "cp", f"{name}:/tests/.", str(readback)])
        if _directory_content_hash(readback) != source_hash:
            raise RecoveryError("container verifier materialization drifted")
    _emit(
        sink,
        "financial_semantic_recovery_verifier_materialized_v1",
        {
            "work_unit_hash": work.work_unit_hash,
            "request_hash": work.request.request_hash,
            "tests_content_hash": source_hash,
            "tests_present_during_agent": False,
            "materialized_after_agent_exit": True,
        },
    )
    started_monotonic = time.monotonic()
    try:
        completed = _run(
            ["docker", "exec", name, "sh", "-lc", profile.verifier_command],
            check=False,
            timeout=1800,
        )
    except subprocess.TimeoutExpired:
        completed = subprocess.CompletedProcess([], -1, "", "timeout")
    duration = round(time.monotonic() - started_monotonic, 6)
    (verifier_dir / "recovery.stdout.txt").write_text(
        completed.stdout or "", encoding="utf-8"
    )
    (verifier_dir / "recovery.stderr.txt").write_text(
        completed.stderr or "", encoding="utf-8"
    )
    reward_path = verifier_dir / "reward.txt"
    ctrf_path = verifier_dir / "ctrf.json"
    if not reward_path.is_file() or not ctrf_path.is_file():
        raise RecoveryError("offline verifier did not persist complete evidence")
    raw_reward = reward_path.read_text(encoding="utf-8").strip()
    if raw_reward not in {"0", "1"}:
        raise RecoveryError("offline verifier reward is malformed")
    body = {
        **{key: value for key, value in started.items() if key not in {"status", "stage_hash"}},
        "status": "completed",
        "completed_at_utc": _utc_now(),
        "verifier_exit": completed.returncode,
        "verifier_duration_seconds": duration,
        "reward": int(raw_reward),
        "reward_sha256": _sha256(reward_path),
        "ctrf_sha256": _sha256(ctrf_path),
        "stdout_sha256": _sha256(verifier_dir / "recovery.stdout.txt"),
        "stderr_sha256": _sha256(verifier_dir / "recovery.stderr.txt"),
        "offline_evaluation_only": True,
        "model_calls": 0,
        "online_judge_calls": 0,
    }
    receipt = _self_hashed(body, "stage_hash")
    _atomic_json(stage_path, receipt)
    _emit(
        sink,
        "financial_semantic_recovery_verifier_completed_v1",
        {
            "work_unit_hash": work.work_unit_hash,
            "request_hash": work.request.request_hash,
            "verifier_exit": completed.returncode,
            "reward": int(raw_reward),
            "stage_hash": receipt["stage_hash"],
            "offline_evaluation_only": True,
        },
    )
    return receipt


def _semantic_resume(
    *,
    context: Mapping[str, Any],
    manifest: Mapping[str, Any],
    minilm_snapshot_root: Path,
    qa_snapshot_root: Path,
    sink: JsonlEventSink,
    common_backend: Mapping[str, Any],
) -> tuple[dict[str, Any], FinancialSemanticSubprocessBackendV1]:
    work = next(work for work in context["plan"].physical_work_units if work.arm == "semantic")
    trial_path = _trial_path(context, work)
    stage_path = trial_path / "semantic_runtime" / SEMANTIC_STAGE
    planner = SharedFinancialSemanticPlannerV1(
        asset_path=context["project"] / context["treatment"].operator_asset_path,
        minilm_runtime_asset_path=context["project"] / context["treatment"].minilm_runtime_asset_path,
        minilm_snapshot_root=minilm_snapshot_root,
        qa_runtime_asset_path=context["project"] / context["treatment"].qa_runtime_asset_path,
        qa_snapshot_root=qa_snapshot_root,
    )
    instruction_path = (
        context["benchmark_root"] / "tasks" / work.family / work.item_id / "instruction.md"
    )
    plan, extraction = planner.build(instruction_path.read_text(encoding="utf-8"))
    binding = manifest["snapshot"]["semantic_pre_agent_binding"]
    if (
        plan.get("plan_hash") != binding["plan_hash"]
        or extraction.get("receipt_hash") != binding["extraction_receipt_hash"]
        or plan.get("instruction_sha256") != binding["instruction_sha256"]
        or work.request.request_hash != binding["request_hash"]
    ):
        raise RecoveryError("rebuilt semantic plan differs from pre-agent binding")
    backend = FinancialSemanticSubprocessBackendV1(
        context["benchmark_root"],
        planner=planner,
        expected_program_id=context["treatment"].recipe_id,
        expected_treatment_hash=context["treatment"].treatment_hash,
        expected_external_skill_source_receipt_hash=(
            context["treatment"].external_skill_source_receipt_hash
        ),
        trials_dir=context["trials_root"] / work.work_unit_hash,
        event_sink=sink,
        **common_backend,
    )
    if stage_path.exists():
        stage = _read_json(stage_path)
        if stage.get("status") != "completed":
            raise RecoveryError("ambiguous prior semantic recovery stage")
        evidence = stage.get("runtime_evidence")
        if not isinstance(evidence, dict) or evidence.get("plan_hash") != plan["plan_hash"]:
            raise RecoveryError("persisted semantic recovery evidence drifted")
        return evidence, backend
    name = _container_name(work)
    if _container_exists(name, "/tests"):
        raise RecoveryError("candidate tests materialized before semantic operator")
    installed = _reconstruct_installed_candidate_receipt(context, name)
    started = _self_hashed(
        {
            "stage_version": RECOVERY_VERSION,
            "status": "started",
            "started_at_utc": _utc_now(),
            "request_hash": work.request.request_hash,
            "plan_hash": plan["plan_hash"],
            "extraction_receipt_hash": extraction["receipt_hash"],
            "installed_candidate_receipt": installed,
            "executed_after_agent_exit": True,
            "executed_before_verifier_materialization": True,
            "model_calls": 0,
            "online_calls": 0,
        },
        "stage_hash",
    )
    _atomic_json(stage_path, started, refuse=True)
    backend._financial_local.state = _FinancialRunStateV1(
        request_hash=work.request.request_hash,
        plan=plan,
        extraction_receipt=extraction,
    )
    backend._execute_financial_plan_before_verifier_v1(
        delegate=subprocess,
        container_name=name,
    )
    state = backend._financial_local.state
    if not isinstance(state, _FinancialRunStateV1) or not isinstance(
        state.runtime_evidence, Mapping
    ):
        raise RecoveryError("semantic runtime evidence was not produced")
    evidence = dict(state.runtime_evidence)
    completed = _self_hashed(
        {
            **{key: value for key, value in started.items() if key not in {"status", "stage_hash"}},
            "status": "completed",
            "completed_at_utc": _utc_now(),
            "runtime_evidence": evidence,
            "runtime_evidence_hash": evidence["evidence_hash"],
        },
        "stage_hash",
    )
    _atomic_json(stage_path, completed)
    return evidence, backend


def _base_result(work: Any, stage: Mapping[str, Any]) -> dict[str, Any]:
    reward = int(stage["reward"])
    return {
        "task_id": f"{work.family}/{work.item_id}",
        "trial_name": f"{work.family}/{work.item_id}/{_skill_config(work)}/{work.request.trial_id}",
        "trial_id": work.request.trial_id,
        "agent": work.request.agent_id,
        "model": work.request.model,
        "skill_config": _skill_config(work),
        "skill_source_dir": None,
        "passed": reward == 1,
        "reward": reward,
        "agent_exit": 0,
        "agent_timed_out": False,
        "verifier_exit": int(stage["verifier_exit"]),
        "agent_stdout": "",
        "agent_stderr": "",
        "token_usage": {},
        "token_usage_source": "codex_action_budget_receipt",
        "scheduler_loss_recovered": True,
    }


def _observation_for_work(
    *,
    context: Mapping[str, Any],
    work: Any,
    backend: SkillLearnSubprocessBackend,
    prebuilt_image: Any,
    verifier_runtime: Any,
    result: Mapping[str, Any],
    duration_seconds: float,
) -> Any:
    audited = backend._audit_trial_artifacts(
        runner=backend._load_runner(),
        request=work.request,
        skill_config=_skill_config(work),
        result=result,
        offline_verifier_profile=verifier_runtime.profile,
        trace_id=f"financial-recovery:{work.work_unit_hash[:20]}",
    )
    observation = backend._sanitize_result(
        work.request,
        result=audited,
        return_code=0 if bool(audited.get("passed")) else 1,
        duration_seconds=duration_seconds,
        prebuilt_image=prebuilt_image,
        offline_verifier_runtime=verifier_runtime,
    )
    return observation, audited


def recover(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project_root.resolve(strict=True)
    batch_root = args.batch_root.resolve(strict=True)
    context = _load_context(
        project=project,
        batch_root=batch_root,
        treatment_manifest=args.treatment_manifest,
    )
    manifest = _verify_prereg(context, args.prereg_manifest.resolve(strict=True))
    _configure_nonsecret_environment(context["protocol"])
    session_path = batch_root / SESSION_RECEIPT
    if session_path.exists():
        session = _read_json(session_path)
        if session.get("preregistration_manifest_hash") != manifest["manifest_hash"]:
            raise RecoveryError("recovery session belongs to another preregistration")
    else:
        current_snapshot = _prereg_snapshot(context)
        if current_snapshot != manifest["snapshot"]:
            raise RecoveryError("orphan state changed after preregistration")
        session = _self_hashed(
            {
                "session_version": RECOVERY_VERSION,
                "started_at_utc": _utc_now(),
                "preregistration_manifest_hash": manifest["manifest_hash"],
                "model_calls_authorized": 0,
                "online_evaluation_authorized": False,
            },
            "session_hash",
        )
        _atomic_json(session_path, session, refuse=True)
    sink = JsonlEventSink(batch_root / RECOVERY_EVENTS)
    _emit(
        sink,
        "financial_semantic_scheduler_loss_recovery_started_v1",
        {
            "session_hash": session["session_hash"],
            "preregistration_manifest_hash": manifest["manifest_hash"],
            "model_calls_authorized": 0,
            "offline_verifier_workers": EXPECTED_VERIFIER_WORKERS,
        },
    )

    validation_root = batch_root / "recovery_asset_validation"
    validation_root.mkdir(exist_ok=True)
    assets = _prepare_runtime_assets_v1(
        project=project,
        destination=validation_root,
        benchmark_root=context["benchmark_root"],
        protocol=context["protocol"],
        split=context["split"],
        event_sink=sink,
        task_input_cache_root=None,
    )
    if (
        assets.preflight_report["report_hash"]
        != manifest["snapshot"]["original_preflight_report_hash"]
    ):
        raise RecoveryError("offline runtime asset preflight changed")
    common_backend = {
        "agent_id": str(context["protocol"].payload["agent_id"]),
        "model": str(context["protocol"].payload["model"]),
        "max_steps": int(context["protocol"].payload["max_steps"]),
        "provider_mode": str(context["protocol"].payload["trial_provider_mode"]),
        "record_upstream": True,
        "prebuilt_cache": assets.prebuilt_cache,
        "offline_verifier_cache": assets.offline_cache,
        "provider_circuit": assets.provider_circuit,
        "model_inference_limiter": assets.model_limiter,
        "codex_agent_execution_policy": context["protocol"].codex_agent_execution_policy,
    }
    semantic_evidence, semantic_backend = _semantic_resume(
        context=context,
        manifest=manifest,
        minilm_snapshot_root=args.minilm_snapshot_root.resolve(strict=True),
        qa_snapshot_root=args.qa_snapshot_root.resolve(strict=True),
        sink=sink,
        common_backend=common_backend,
    )

    orphan_works = [
        work
        for work in context["plan"].physical_work_units
        if not (_trial_path(context, work) / "result.json").is_file()
    ]
    if len(orphan_works) != EXPECTED_ORPHANS:
        raise RecoveryError("recovery orphan count changed")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=EXPECTED_VERIFIER_WORKERS
    ) as executor:
        futures = {
            executor.submit(
                _copy_tests_and_verify,
                context=context,
                work=work,
                sink=sink,
            ): work
            for work in orphan_works
        }
        verifier_stages = {
            futures[future].work_unit_hash: future.result() for future in futures
        }

    original_completion_events = {
        str(row["payload"]["request_hash"]): row["payload"]
        for row in _event_rows(batch_root / ORIGINAL_EVENTS)
        if row.get("event") == "skilllearn_trial_completed"
    }
    installed_receipt = _read_json(
        _trial_path(
            context,
            next(work for work in context["plan"].physical_work_units if work.arm == "semantic"),
        )
        / "semantic_runtime"
        / SEMANTIC_STAGE
    )["installed_candidate_receipt"]
    physical_rows = []
    observations: dict[str, Any] = {}
    for work in context["plan"].physical_work_units:
        trial_path = _trial_path(context, work)
        result_path = trial_path / "result.json"
        stage = verifier_stages.get(work.work_unit_hash)
        if stage is None:
            result = _read_json(result_path)
            duration = float(
                original_completion_events[work.request.request_hash]["duration_seconds"]
            )
            recovery_kind = "original_precompleted"
        else:
            result = _base_result(work, stage)
            if work.arm == "semantic":
                result["installed_skill_source_receipt_hash"] = (
                    context["treatment"].external_skill_source_receipt_hash
                )
                result["installed_skill_tree_hash"] = installed_receipt[
                    "installed_tree_hash"
                ]
                result["installed_skill_destination_count"] = installed_receipt[
                    "destination_count"
                ]
            _atomic_json(result_path, result)
            duration = 0.0
            recovery_kind = "post_agent_offline_verifier_recovered"
        if work.arm == "semantic":
            backend = semantic_backend
        else:
            backend = SkillLearnSubprocessBackend(
                context["benchmark_root"],
                trials_dir=context["trials_root"] / work.work_unit_hash,
                event_sink=sink,
                **common_backend,
            )
        image, verifier_runtime = backend.prewarm_trial_environment(
            family=work.family,
            item_id=work.item_id,
            trace_id=f"financial-recovery-asset:{work.work_unit_hash[:20]}",
        )
        observation, audited = _observation_for_work(
            context=context,
            work=work,
            backend=backend,
            prebuilt_image=image,
            verifier_runtime=verifier_runtime,
            result=result,
            duration_seconds=duration,
        )
        if not observation.valid:
            raise RecoveryError("recovered physical observation is invalid")
        if stage is None and observation.observation_hash != original_completion_events[
            work.request.request_hash
        ]["observation_hash"]:
            raise RecoveryError("precompleted observation could not be reproduced")
        observations[work.work_unit_hash] = observation
        physical_rows.append(
            {
                **work.safe_payload(),
                "observation": observation.to_dict(),
                "observation_hash": observation.observation_hash,
                "recovery_kind": recovery_kind,
                "duration_reconstructed": stage is None,
                "verifier_stage_hash": stage["stage_hash"] if stage else None,
                "semantic_runtime_evidence_hash": (
                    semantic_evidence["evidence_hash"] if work.arm == "semantic" else None
                ),
                "audited_upstream_result_hash": observation.upstream_result_hash,
            }
        )

    raw_by_item = {
        work.item_id: observations[work.work_unit_hash]
        for work in context["plan"].physical_work_units
        if work.arm == "raw"
    }
    semantic_work = next(
        work for work in context["plan"].physical_work_units if work.arm == "semantic"
    )
    semantic_observation = observations[semantic_work.work_unit_hash]
    active_raw = raw_by_item[ACTIVE_FRESH_ITEM_ID]
    projections = []
    for item_id in context["split"].item_ids:
        if item_id == ACTIVE_FRESH_ITEM_ID:
            continue
        raw = raw_by_item[item_id]
        projected = raw.as_variant(context["plan"].candidate_requests_by_item[item_id])
        projections.append(
            {
                "projection_policy": "exact_raw_inactive_route_projection_v1",
                "item_id_hash": stable_hash({"item_id": item_id}),
                "raw_observation_hash": raw.observation_hash,
                "candidate_request_hash": projected.request.request_hash,
                "projected_observation_hash": projected.observation_hash,
                "raw_success": raw.success,
                "projected_success": projected.success,
                "behavior_identical_by_predeclared_inactive_route": True,
                "model_calls": 0,
            }
        )

    artifact_closure = _worker_artifact_closure(context["trials_root"])
    cleanup_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=EXPECTED_ORPHANS) as executor:
        removal = {
            executor.submit(_run, ["docker", "rm", "-f", _container_name(work)]): work
            for work in orphan_works
        }
        for future, work in removal.items():
            future.result()
            cleanup_rows.append(
                {
                    "work_unit_hash": work.work_unit_hash,
                    "container_name_hash": stable_hash(
                        {"container_name": _container_name(work)}
                    ),
                    "removed": True,
                }
            )

    active_gain = int(semantic_observation.success) - int(active_raw.success)
    body = {
        "report_version": RECOVERY_VERSION,
        "execution_completed": True,
        "evidence_valid": True,
        "scheduler_process_loss": True,
        "pristine_runner_completion": False,
        "post_agent_recovery_disclosed": True,
        "preregistration_manifest_hash": manifest["manifest_hash"],
        "recovery_session_hash": session["session_hash"],
        "treatment_manifest_hash": context["treatment"].manifest_hash,
        "fresh_split_manifest_hash": context["split"].manifest_hash,
        "execution_plan_hash": context["plan"].plan_hash,
        "physical_results": sorted(physical_rows, key=lambda row: row["work_unit_hash"]),
        "physical_result_set_hash": stable_hash(
            {"rows": sorted(physical_rows, key=lambda row: row["work_unit_hash"])}
        ),
        "inactive_projections": sorted(projections, key=lambda row: row["item_id_hash"]),
        "active_pair": {
            "item_id_hash": stable_hash({"item_id": ACTIVE_FRESH_ITEM_ID}),
            "raw_observation_hash": active_raw.observation_hash,
            "candidate_observation_hash": semantic_observation.observation_hash,
            "raw_success": active_raw.success,
            "candidate_success": semantic_observation.success,
            "candidate_minus_raw": active_gain,
            "raw_error_type": active_raw.error_type,
            "candidate_error_type": semantic_observation.error_type,
        },
        "paired_task_utility_measurement_valid": True,
        "causal_measurement_status": "valid_preregistered_post_agent_resume",
        "financial_runtime_evidence": [semantic_evidence],
        "financial_runtime_evidence_set_hash": stable_hash(
            {"rows": [semantic_evidence]}
        ),
        "worker_artifact_closure": artifact_closure,
        "original_model_execution_count": EXPECTED_PHYSICAL,
        "replayed_model_execution_count": 0,
        "recovery_model_call_count": 0,
        "offline_verifier_recovery_count": EXPECTED_ORPHANS,
        "offline_verifier_worker_count": EXPECTED_VERIFIER_WORKERS,
        "offline_evaluation_only": True,
        "online_judge_calls": 0,
        "network_fallback_used": False,
        "selected_provider_unchanged": True,
        "official_hipporag": False,
        "hipporag_status": OFFICIAL_HIPPORAG_STATUS,
        "official_hipporag_execution_count": 0,
        "new_performance_gate_added": False,
        "promotion_gate_applied": False,
        "promotion_authorized": False,
        "sealed_test_accessed": False,
        "recovered_duration_values_available": False,
        "duration_or_cost_comparison_authorized": False,
        "container_cleanup": sorted(cleanup_rows, key=lambda row: row["work_unit_hash"]),
        "secret_value_persisted": False,
    }
    report = _self_hashed(body, "report_hash")
    _atomic_json(batch_root / RECOVERY_REPORT, report, refuse=True)
    _emit(
        sink,
        "financial_semantic_scheduler_loss_recovery_completed_v1",
        {
            "report_hash": report["report_hash"],
            "raw_success": active_raw.success,
            "candidate_success": semantic_observation.success,
            "candidate_minus_raw": active_gain,
            "model_calls": 0,
            "offline_evaluation_only": True,
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--treatment-manifest", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prereg = subparsers.add_parser("preregister")
    prereg.add_argument("--output", type=Path, required=True)
    execute = subparsers.add_parser("recover")
    execute.add_argument("--prereg-manifest", type=Path, required=True)
    execute.add_argument("--minilm-snapshot-root", type=Path, required=True)
    execute.add_argument("--qa-snapshot-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = preregister(args) if args.command == "preregister" else recover(args)
    except Exception as error:
        if args.command == "recover":
            failure_body = {
                "report_version": RECOVERY_VERSION,
                "execution_completed": False,
                "error_type": type(error).__name__,
                "error_message_hash": stable_hash({"message": str(error)}),
                "model_replay_attempted": False,
                "secret_value_persisted": False,
            }
            try:
                _atomic_json(
                    args.batch_root.resolve() / RECOVERY_FAILURE,
                    _self_hashed(failure_body, "report_hash"),
                )
            except Exception:
                pass
        raise
    print(
        json.dumps(
            {
                "completed": True,
                "command": args.command,
                "hash": result.get("report_hash") or result.get("manifest_hash"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
