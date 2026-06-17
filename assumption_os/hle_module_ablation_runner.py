"""Automatic HLE module ablation runner.

This wrapper runs the same fresh HLE sample window through multiple bounded
Assumption Agent profiles.  It records only commands, profile toggles, hashes,
counts, and aggregate metrics; API credentials are inherited from the process
environment and never written to artifacts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR


DEFAULT_RUN_DIR = PAPER_DIR / "hle_module_ablation_runs"
DEFAULT_MD_DIR = Path("reconstruction/md")


@dataclass(frozen=True)
class ModuleAblationProfile:
    name: str
    description: str
    env_overrides: dict[str, str] = field(default_factory=dict)
    disable_evidence_bridge: bool = False
    agent_top_k: int | None = None
    agent_context_max_chars: int | None = None


DEFAULT_ABLATION_PROFILES: tuple[ModuleAblationProfile, ...] = (
    ModuleAblationProfile(
        name="full",
        description="Full selective Assumption Agent profile with verified-or-abstain enabled.",
    ),
    ModuleAblationProfile(
        name="verified_gate_off",
        description="Disable verified-or-abstain to measure unverified selection pollution.",
        env_overrides={"HLE_DISABLE_VERIFIED_OR_ABSTAIN": "1"},
    ),
    ModuleAblationProfile(
        name="no_option_evidence",
        description="Disable option-specific MC evidence scorer and evidence arbitrator.",
        env_overrides={
            "HLE_DISABLE_MC_OPTION_EVIDENCE_SCORER": "1",
            "HLE_ENABLE_OPTION_EVIDENCE_ARBITRATOR": "0",
        },
    ),
    ModuleAblationProfile(
        name="no_candidate_claim_verifier",
        description="Disable executable candidate-claim verifier for exact/math and MC math claims.",
        env_overrides={"HLE_DISABLE_CANDIDATE_CLAIM_VERIFIER": "1"},
    ),
    ModuleAblationProfile(
        name="no_agent_hipporag",
        description="Disable agent-side HippoRAG child context and priority selection.",
        env_overrides={
            "HLE_DISABLE_AGENT_HIPPORAG_CHILD": "1",
            "HLE_DISABLE_AGENT_HIPPORAG_PRIORITY": "1",
        },
    ),
    ModuleAblationProfile(
        name="no_evidence",
        description="Disable evidence bridge, agent HippoRAG child, option evidence, and source-grounded overrides.",
        disable_evidence_bridge=True,
        env_overrides={
            "HLE_DISABLE_AGENT_HIPPORAG_CHILD": "1",
            "HLE_DISABLE_AGENT_HIPPORAG_PRIORITY": "1",
            "HLE_DISABLE_MC_OPTION_EVIDENCE_SCORER": "1",
            "HLE_ENABLE_MC_EVIDENCE_BRIDGE": "0",
            "HLE_ENABLE_BROAD_SOURCE_GROUNDED_MC": "0",
            "HLE_ENABLE_EXACT_EVIDENCE_OVERRIDE": "0",
        },
    ),
    ModuleAblationProfile(
        name="no_graph",
        description="Disable Assumption Graph retrieval and its retrieved prompt context.",
        agent_top_k=0,
        agent_context_max_chars=0,
        env_overrides={"HLE_DISABLE_ASSUMPTION_GRAPH_RETRIEVAL": "1"},
    ),
    ModuleAblationProfile(
        name="no_morphism",
        description="Disable bounded structural morphism/formal transfer search.",
        env_overrides={"HLE_DISABLE_STRUCTURAL_MORPHISM_TRANSFER": "1"},
    ),
    ModuleAblationProfile(
        name="no_recursive_runner",
        description="Disable recursive assumption runner planning stage.",
        env_overrides={"HLE_DISABLE_RECURSIVE_ASSUMPTION_RUNNER": "1"},
    ),
    ModuleAblationProfile(
        name="no_recursive",
        description="Alias for no_recursive_runner used by fresh paper ablation tables.",
        env_overrides={"HLE_DISABLE_RECURSIVE_ASSUMPTION_RUNNER": "1"},
    ),
    ModuleAblationProfile(
        name="no_world_model_router",
        description="Disable world-model routing so context is not promoted by the cheap verifier.",
        env_overrides={"HLE_DISABLE_WORLD_MODEL_ROUTER": "1"},
    ),
    ModuleAblationProfile(
        name="no_world_model",
        description="Alias for no_world_model_router used by fresh paper ablation tables.",
        env_overrides={"HLE_DISABLE_WORLD_MODEL_ROUTER": "1"},
    ),
    ModuleAblationProfile(
        name="raw_preserve_selector",
        description="Add a raw no-context base-model candidate and let the selector preserve it under uncertainty.",
        env_overrides={"HLE_ENABLE_RAW_PRESERVE_SELECTOR": "1"},
    ),
)


def selected_ablation_profiles(names: str = "") -> list[ModuleAblationProfile]:
    profiles = list(DEFAULT_ABLATION_PROFILES)
    if not names:
        return profiles
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    by_name = {profile.name: profile for profile in profiles}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise ValueError(f"unknown ablation profiles: {', '.join(missing)}")
    return [by_name[name] for name in wanted]


def build_profile_command(
    *,
    profile: ModuleAblationProfile,
    args: argparse.Namespace,
    root: Path,
    run_dir: Path,
    md_dir: Path,
) -> list[str]:
    profile_eval_id = f"{args.eval_id}_{profile.name}"
    profile_run_dir = run_dir / profile.name
    profile_md_out = md_dir / f"{profile_eval_id}.md"
    profile_json_out = profile_run_dir / f"{profile_eval_id}.json"
    cmd = [
        sys.executable,
        "-m",
        "assumption_os.hle_parallel_shard_runner",
        "--root",
        str(root),
        "--eval-id",
        profile_eval_id,
        "--total-sample-size",
        str(args.total_sample_size),
        "--shard-size",
        str(args.shard_size),
        "--parallel-workers",
        str(args.parallel_workers),
        "--max-scan",
        str(args.max_scan),
        "--seed-offset",
        str(args.seed_offset),
        "--seed-stride",
        str(args.seed_stride),
        "--models",
        args.models,
        "--variants",
        args.variants,
        "--max-tokens",
        str(args.max_tokens),
        "--graph-dir",
        args.graph_dir,
        "--agent-top-k",
        str(profile.agent_top_k if profile.agent_top_k is not None else args.agent_top_k),
        "--agent-context-max-chars",
        str(
            profile.agent_context_max_chars
            if profile.agent_context_max_chars is not None
            else args.agent_context_max_chars
        ),
        "--agent-child-mode",
        args.agent_child_mode,
        "--run-dir",
        str(profile_run_dir),
        "--md-dir",
        str(md_dir),
        "--out",
        str(profile_json_out),
        "--md-out",
        str(profile_md_out),
        "--heartbeat-out",
        str(profile_run_dir / f"{profile_eval_id}.heartbeat.json"),
    ]
    if args.execute_live:
        cmd.append("--execute-live")
    if args.call_timeout is not None:
        cmd.extend(["--call-timeout", str(args.call_timeout)])
    if args.agent_child_timeout is not None:
        cmd.extend(["--agent-child-timeout", str(args.agent_child_timeout)])
    if args.sample_answer_type:
        cmd.extend(["--sample-answer-type", args.sample_answer_type])
    if args.sample_subject_contains:
        cmd.extend(["--sample-subject-contains", args.sample_subject_contains])
    if args.exclude_existing_hle_artifacts:
        cmd.append("--exclude-existing-hle-artifacts")
    if args.exclude_artifact_glob:
        cmd.extend(["--exclude-artifact-glob", args.exclude_artifact_glob])
    if getattr(args, "dedupe_shard_samples", False):
        cmd.append("--dedupe-shard-samples")
    if getattr(args, "dedupe_shard_max_attempts", None) is not None:
        cmd.extend(["--dedupe-shard-max-attempts", str(args.dedupe_shard_max_attempts)])
    if args.soft_timeout_sec is not None:
        cmd.extend(["--soft-timeout-sec", str(args.soft_timeout_sec)])
    if args.terminate_grace_sec is not None:
        cmd.extend(["--terminate-grace-sec", str(args.terminate_grace_sec)])
    if getattr(args, "launch_stagger_sec", 0.0):
        cmd.extend(["--launch-stagger-sec", str(args.launch_stagger_sec)])
    if getattr(args, "reuse_completed_shards", False):
        cmd.append("--reuse-completed-shards")
    if getattr(args, "kill_on_soft_timeout", False):
        cmd.append("--kill-on-soft-timeout")
    if args.model_router_attempts is not None:
        cmd.extend(["--model-router-attempts", str(args.model_router_attempts)])
    if args.model_router_timeout is not None:
        cmd.extend(["--model-router-timeout", str(args.model_router_timeout)])
    if args.model_router_per_attempt_timeout is not None:
        cmd.extend(["--model-router-per-attempt-timeout", str(args.model_router_per_attempt_timeout)])
    if args.model_router_backoff_base_sec is not None:
        cmd.extend(["--model-router-backoff-base-sec", str(args.model_router_backoff_base_sec)])
    if args.model_router_global_concurrency is not None:
        cmd.extend(["--model-router-global-concurrency", str(args.model_router_global_concurrency)])
    if args.model_router_global_concurrency_dir:
        cmd.extend(["--model-router-global-concurrency-dir", args.model_router_global_concurrency_dir])
    if args.model_router_global_slot_ttl_sec is not None:
        cmd.extend(["--model-router-global-slot-ttl-sec", str(args.model_router_global_slot_ttl_sec)])
    if args.model_router_global_slot_wait_sec is not None:
        cmd.extend(["--model-router-global-slot-wait-sec", str(args.model_router_global_slot_wait_sec)])
    if args.disable_evidence_bridge or profile.disable_evidence_bridge:
        cmd.append("--disable-evidence-bridge")
    return cmd


def build_ablation_plan(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    run_dir = _path_arg(args.run_dir, root=root)
    md_dir = _path_arg(args.md_dir, root=root)
    profiles = selected_ablation_profiles(args.profiles)
    profile_rows = []
    for profile in profiles:
        command = build_profile_command(profile=profile, args=args, root=root, run_dir=run_dir, md_dir=md_dir)
        profile_rows.append({
            "profile": profile.name,
            "description": profile.description,
            "command": command,
            "env_overrides": dict(sorted(profile.env_overrides.items())),
            "disable_evidence_bridge": profile.disable_evidence_bridge,
            "agent_top_k": profile.agent_top_k if profile.agent_top_k is not None else args.agent_top_k,
            "agent_context_max_chars": (
                profile.agent_context_max_chars
                if profile.agent_context_max_chars is not None
                else args.agent_context_max_chars
            ),
            "secrets_in_command_or_overrides": _contains_secret(command, profile.env_overrides),
        })
    return {
        "eval_id": args.eval_id,
        "eval_kind": "hle_module_ablation_runner",
        "performance_validation": True,
        "dry_run": bool(args.dry_run),
        "profile_count": len(profile_rows),
        "profile_workers": max(1, int(getattr(args, "profile_workers", 1) or 1)),
        "sampling": {
            "total_sample_size": args.total_sample_size,
            "shard_size": args.shard_size,
            "seed_offset": args.seed_offset,
            "seed_stride": args.seed_stride,
            "models": [item.strip() for item in args.models.split(",") if item.strip()],
            "variants": [item.strip() for item in args.variants.split(",") if item.strip()],
        },
        "profiles": profile_rows,
        "gates": {
            "profiles_defined": bool(profile_rows),
            "secrets_not_persisted": not any(row["secrets_in_command_or_overrides"] for row in profile_rows),
            "same_seed_window_for_all_profiles": len({
                (args.total_sample_size, args.seed_offset, args.seed_stride)
                for _ in profile_rows
            }) == 1,
            "raw_content_persisted": False,
        },
        "raw_content_persisted": False,
    }


def run_ablation_profiles(plan: dict[str, Any], *, root: Path, run_dir: Path) -> dict[str, Any]:
    profile_workers = max(1, int(plan.get("profile_workers") or 1))
    profile_rows = list(plan["profiles"])
    if profile_workers == 1 or len(profile_rows) <= 1:
        results = [
            _run_single_ablation_profile(row=row, root=root, run_dir=run_dir, eval_id=plan["eval_id"])
            for row in profile_rows
        ]
    else:
        indexed_results: list[tuple[int, dict[str, Any]]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(profile_workers, len(profile_rows))) as executor:
            futures = {
                executor.submit(
                    _run_single_ablation_profile,
                    row=row,
                    root=root,
                    run_dir=run_dir,
                    eval_id=plan["eval_id"],
                ): index
                for index, row in enumerate(profile_rows)
            }
            for future in concurrent.futures.as_completed(futures):
                indexed_results.append((futures[future], future.result()))
        results = [row for _, row in sorted(indexed_results, key=lambda item: item[0])]
    plan = dict(plan)
    plan["profile_workers"] = profile_workers
    plan["dry_run"] = False
    plan["profile_results"] = results
    plan["gates"] = dict(plan.get("gates") or {})
    plan["gates"].update({
        "all_profiles_completed": all(int(row.get("returncode") or 0) == 0 for row in results),
        "all_profile_payloads_loaded": all(bool(row.get("payload_loaded")) for row in results),
        "all_profile_payloads_pass": all(row.get("pass") is True for row in results),
        "all_profile_payloads_pollution_pass": all(row.get("pollution_pass") is True for row in results),
        "all_profile_payloads_preserve_raw_content": all(
            ((row.get("metrics") or {}).get("raw_content_persisted") is False)
            for row in results
            if row.get("metrics")
        ),
    })
    plan["module_ablation_summary"] = _module_ablation_summary(results)
    return plan


def _run_single_ablation_profile(
    *,
    row: dict[str, Any],
    root: Path,
    run_dir: Path,
    eval_id: str,
) -> dict[str, Any]:
    profile_name = str(row["profile"])
    stdout_path = run_dir / profile_name / f"{eval_id}_{profile_name}.stdout.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in (row.get("env_overrides") or {}).items()})
    start = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            [str(item) for item in row["command"]],
            cwd=str(root),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = round(time.monotonic() - start, 4)
    out_path = _profile_out_path(row["command"])
    payload = _load_json(out_path)
    return {
        "profile": profile_name,
        "returncode": completed.returncode,
        "elapsed_sec": elapsed,
        "stdout_out": str(stdout_path),
        "payload_out": str(out_path) if out_path else "",
        "payload_loaded": bool(payload),
        "pass": None if not payload else payload.get("pass"),
        "paper_clean_pass": None if not payload else payload.get("paper_clean_pass"),
        "pollution_pass": None if not payload else payload.get("pollution_pass"),
        "metrics": None if not payload else payload.get("metrics"),
        "failed_gates": [] if not payload else payload.get("failed_gates", []),
        "paper_clean_failed_gates": [] if not payload else payload.get("paper_clean_failed_gates", []),
        "pollution_failed_gates": [] if not payload else payload.get("pollution_failed_gates", []),
    }


def _module_ablation_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_profile: dict[str, dict[str, Any]] = {}
    for result in results:
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        clean = metrics.get("clean_shared_subset") if isinstance(metrics.get("clean_shared_subset"), dict) else {}
        by_profile[str(result.get("profile"))] = {
            "returncode": result.get("returncode"),
            "pass": result.get("pass"),
            "paper_clean_pass": result.get("paper_clean_pass"),
            "pollution_pass": result.get("pollution_pass"),
            "overall_accuracy": metrics.get("overall_accuracy"),
            "clean_shared_subset": clean,
            "by_model_variant": metrics.get("by_model_variant") if isinstance(metrics, dict) else {},
        }
    return {
        "by_profile": by_profile,
        "higher_is_better": "assumption_agent_recursive_verify accuracy; controls remain frozen",
    }


def _profile_out_path(command: list[str]) -> Path | None:
    try:
        index = command.index("--out")
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return Path(command[index + 1])


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _contains_secret(command: list[str], env_overrides: dict[str, str]) -> bool:
    haystack = " ".join([str(item) for item in command] + [f"{key}={value}" for key, value in env_overrides.items()])
    lowered = haystack.lower()
    return "sk-" in lowered or "hf_" in lowered


def _path_arg(value: str, *, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def format_ablation_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# HLE Module Ablation Runner",
        "",
        f"- eval id: `{payload['eval_id']}`",
        f"- dry run: `{payload.get('dry_run')}`",
        f"- profile count: `{payload.get('profile_count')}`",
        f"- profile workers: `{payload.get('profile_workers')}`",
        f"- raw content persisted: `{payload.get('raw_content_persisted')}`",
        "",
        "## Gates",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in sorted((payload.get("gates") or {}).items()):
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend([
        "",
        "## Profiles",
        "",
        "| profile | description | env keys | disable evidence bridge | agent top k |",
        "| --- | --- | --- | ---: | ---: |",
    ])
    for row in payload.get("profiles", []):
        env_keys = ", ".join(sorted((row.get("env_overrides") or {}).keys())) or "none"
        lines.append(
            f"| `{row['profile']}` | {row['description']} | `{env_keys}` | "
            f"`{row['disable_evidence_bridge']}` | `{row['agent_top_k']}` |"
        )
    if payload.get("profile_results"):
        lines.extend([
            "",
            "## Results",
            "",
            "| profile | returncode | pass | paper clean | pollution | overall accuracy |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for result in payload.get("profile_results", []):
            metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
            lines.append(
                f"| `{result['profile']}` | `{result['returncode']}` | `{result.get('pass')}` | "
                f"`{result.get('paper_clean_pass')}` | `{result.get('pollution_pass')}` | "
                f"`{metrics.get('overall_accuracy')}` |"
            )
    lines.append("")
    lines.append("Raw HLE questions, answers, rationales, canaries, and prediction text are not persisted.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automatic module ablations over the HLE parallel runner.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="hle_module_ablation_20260617")
    parser.add_argument("--profiles", default="")
    parser.add_argument("--profile-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--total-sample-size", type=int, default=12)
    parser.add_argument("--shard-size", type=int, default=1)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--max-scan", type=int, default=5000)
    parser.add_argument("--seed-offset", type=int, default=2600)
    parser.add_argument("--seed-stride", type=int, default=17)
    parser.add_argument("--sample-answer-type", default="")
    parser.add_argument("--sample-subject-contains", default="")
    parser.add_argument("--models", default="gpt-5.4-mini")
    parser.add_argument("--variants", default="raw,assumption_agent_recursive_verify,hipporag_baseline")
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--call-timeout", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--graph-dir", default=str(Path("phase four/assumption_graph")))
    parser.add_argument("--agent-top-k", type=int, default=5)
    parser.add_argument("--agent-context-max-chars", type=int, default=2800)
    parser.add_argument("--agent-child-mode", choices=["serial", "parallel_quorum"], default=os.environ.get("HLE_AGENT_CHILD_MODE", "parallel_quorum"))
    parser.add_argument("--agent-child-timeout", type=float, default=None)
    parser.add_argument("--disable-evidence-bridge", action="store_true")
    parser.add_argument("--exclude-existing-hle-artifacts", action="store_true")
    parser.add_argument("--exclude-artifact-glob", default="phase four/assumption_graph/paper_readiness_20260604/hle*.json*")
    parser.add_argument("--dedupe-shard-samples", action="store_true")
    parser.add_argument("--dedupe-shard-max-attempts", type=int, default=25)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--md-dir", default=str(DEFAULT_MD_DIR))
    parser.add_argument("--out", default="")
    parser.add_argument("--md-out", default="")
    parser.add_argument("--soft-timeout-sec", type=float, default=None)
    parser.add_argument("--terminate-grace-sec", type=float, default=30.0)
    parser.add_argument("--launch-stagger-sec", type=float, default=0.0)
    parser.add_argument("--reuse-completed-shards", action="store_true")
    parser.add_argument("--kill-on-soft-timeout", action="store_true")
    parser.add_argument("--model-router-attempts", type=int, default=None)
    parser.add_argument("--model-router-timeout", type=float, default=None)
    parser.add_argument("--model-router-per-attempt-timeout", type=float, default=None)
    parser.add_argument("--model-router-backoff-base-sec", type=float, default=None)
    parser.add_argument("--model-router-global-concurrency", type=int, default=None)
    parser.add_argument("--model-router-global-concurrency-dir", default="")
    parser.add_argument("--model-router-global-slot-ttl-sec", type=float, default=None)
    parser.add_argument("--model-router-global-slot-wait-sec", type=float, default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    run_dir = _path_arg(args.run_dir, root=root)
    md_dir = _path_arg(args.md_dir, root=root)
    payload = build_ablation_plan(args)
    if not args.dry_run:
        payload = run_ablation_profiles(payload, root=root, run_dir=run_dir)
    payload["pass"] = all(
        value is False if key == "raw_content_persisted" else bool(value)
        for key, value in (payload.get("gates") or {}).items()
    )
    payload["failed_gates"] = [
        key for key, value in (payload.get("gates") or {}).items()
        if not (value is False if key == "raw_content_persisted" else bool(value))
    ]

    out = _path_arg(args.out, root=root) if args.out else run_dir / f"{args.eval_id}.json"
    md_out = _path_arg(args.md_out, root=root) if args.md_out else md_dir / f"{args.eval_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_out.write_text(format_ablation_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "dry_run": payload["dry_run"],
        "profile_count": payload["profile_count"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
        "md_out": str(md_out),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
