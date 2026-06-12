"""Paper-facing fresh frozen rerun protocol.

The frozen main paper line is already organized at problem level, and the
blinded recursive line has a 240-call fresh pilot.  This module locks the next
paper-facing fresh rerun protocol: heldout problem pool, seed schedule,
parallel execution plan, baseline/toggle comparisons, redaction policy,
bootstrap CI rules, and claim boundaries.

It intentionally does not execute a large live run.  Passing this artifact
means the fresh protocol is ready and reproducible; it does not claim the
target fresh rerun has already been completed.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash
from .full_v3_blinded_recursive_live_line import (
    DEFAULT_EXISTING_SAMPLES,
    DEFAULT_PROBLEM_DIR,
    build_full_v3_blinded_recursive_live_line_payload,
)
from .paper_frozen_main_experiment_v2 import BASELINE_ARMS


DEFAULT_OUT = PAPER_DIR / "paper_fresh_frozen_rerun_protocol_20260612.json"
DEFAULT_MD_OUT = Path("reconstruction/md/paper_fresh_frozen_rerun_protocol_20260612.md")

SOURCE_ARTIFACTS = {
    "paper_frozen_main": PAPER_DIR / "paper_frozen_main_experiment_v2_20260612.json",
    "fresh_blinded_pilot": PAPER_DIR / "full_v3_blinded_recursive_live_line_20260612.json",
    "open_framework_run": PAPER_DIR / "open_ended_framework_evolution_run_20260612.json",
    "simulator_leakage_audit": PAPER_DIR / "simulator_no_leakage_audit_20260612.json",
    "last_three_coverage": PAPER_DIR / "last_three_part_coverage_audit_20260612.json",
}

DEFAULT_PROTOCOL = {
    "target_fresh_api_call_count": 720,
    "generations": 5,
    "seed_values": [20260612, 20260613, 20260614, 20260615],
    "candidates_per_generation": 4,
    "trigger_rows_per_candidate": 6,
    "control_rows_per_candidate": 3,
    "parallel_workers": 16,
    "bootstrap_samples": 4000,
    "model_alias": "gpt_mini",
}

SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9]{12,}|Bearer\s+[A-Za-z0-9._-]{12,}|newapi_channel_" r"conn)",
    re.IGNORECASE,
)


def build_paper_fresh_frozen_rerun_protocol_payload(
    *,
    root: Path,
    eval_id: str = "paper_fresh_frozen_rerun_protocol_20260612",
    target_fresh_api_call_count: int = DEFAULT_PROTOCOL["target_fresh_api_call_count"],
) -> dict[str, Any]:
    root = root.resolve()
    artifacts = {name: _load_json(root / path) for name, path in SOURCE_ARTIFACTS.items()}
    protocol = dict(DEFAULT_PROTOCOL)
    protocol["target_fresh_api_call_count"] = target_fresh_api_call_count
    problem_pool, problem_report = _fresh_problem_pool(root)
    dry_run = build_full_v3_blinded_recursive_live_line_payload(
        root=root,
        eval_id=f"{eval_id}_dry_run",
        execution_mode="dry_run",
        generations=protocol["generations"],
        seed_values=protocol["seed_values"],
        candidates_per_generation=protocol["candidates_per_generation"],
        trigger_rows_per_candidate=protocol["trigger_rows_per_candidate"],
        control_rows_per_candidate=protocol["control_rows_per_candidate"],
        model_alias=protocol["model_alias"],
        parallel_workers=protocol["parallel_workers"],
        min_planned_calls_for_gate=target_fresh_api_call_count,
        bootstrap_samples=protocol["bootstrap_samples"],
        screen_artifacts=[SOURCE_ARTIFACTS["fresh_blinded_pilot"]],
        load_keyfile=False,
    )
    commands = _commands(protocol=protocol)
    protocol_manifest = _protocol_manifest(
        root=root,
        protocol=protocol,
        problem_pool=problem_pool,
        artifacts=artifacts,
        commands=commands,
    )
    metrics = _metrics(
        artifacts=artifacts,
        protocol=protocol,
        problem_report=problem_report,
        dry_run=dry_run,
        commands=commands,
        protocol_manifest=protocol_manifest,
    )
    gates = {
        "source_artifacts_pass": metrics["source_artifact_pass_rate"] == 1.0,
        "frozen_main_large_and_positive": metrics["frozen_problem_count"] >= 1000
        and metrics["frozen_margin_over_best_baseline"] > 0,
        "fresh_pilot_completed": metrics["fresh_pilot_api_call_count"] >= 180
        and metrics["fresh_pilot_pass"] is True,
        "fresh_pilot_redacted": metrics["fresh_pilot_prompt_answer_or_secret_payload_detected"] is False,
        "target_call_budget_high": metrics["target_fresh_api_call_count"] >= 720,
        "dry_run_matches_target_budget": metrics["dry_run_planned_fresh_api_call_count"]
        == metrics["target_fresh_api_call_count"],
        "dry_run_protocol_passes": metrics["dry_run_pass"] is True,
        "heldout_problem_pool_sufficient": metrics["available_problem_count"] >= metrics["target_fresh_api_call_count"],
        "heldout_problem_domain_coverage": metrics["available_problem_domain_count"] >= 6,
        "problem_manifest_is_redacted": metrics["problem_manifest_raw_payload_exposed"] is False,
        "problem_manifest_disjoint_from_prior_samples": metrics["disjoint_from_existing_samples"] is True,
        "baseline_suite_locked": metrics["baseline_count"] >= 8,
        "ci_plan_problem_level": metrics["bootstrap_samples"] >= 4000,
        "commands_secret_free": metrics["command_secret_hit_count"] == 0,
        "artifact_hashes_locked": metrics["source_artifact_hash_count"] == len(SOURCE_ARTIFACTS),
        "simulator_leakage_audit_passes": metrics["simulator_leakage_audit_pass"] is True,
        "target_result_not_overclaimed": metrics["target_fresh_result_claim_allowed"] is False,
        "protocol_ready_claim_allowed": metrics["fresh_protocol_ready_claim_allowed"] is True,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "paper_fresh_frozen_rerun_protocol",
        "reconstruction_v2_full_phase": "paper_facing_fresh_frozen_rerun_protocol",
        "implementation_level": "fresh_protocol_readiness_not_completed_target_live_run",
        "performance_validation": True,
        "validation_scope": (
            "Locks the paper-facing fresh rerun protocol after the frozen main table and 240-call fresh pilot.  "
            "The artifact freezes problem-pool hashes, seed schedule, parallel execution commands, baseline and "
            "toggle comparisons, redaction rules, and problem-level CI requirements.  It does not claim the "
            "target 720-call fresh rerun has already been executed."
        ),
        "source_artifacts": {
            name: {
                "path": str(path),
                "exists": (root / path).exists(),
                "pass": bool(artifacts[name].get("pass")),
                "eval_kind": artifacts[name].get("eval_kind"),
                "sha256": _sha256(root / path) if (root / path).exists() else None,
            }
            for name, path in SOURCE_ARTIFACTS.items()
        },
        "protocol": protocol,
        "problem_manifest": problem_report,
        "protocol_manifest": protocol_manifest,
        "dry_run_protocol": {
            "eval_id": dry_run["eval_id"],
            "pass": dry_run["pass"],
            "metrics": dry_run["metrics"],
            "problem_level_ci": dry_run["problem_level_ci"],
            "failed_gates": dry_run["failed_gates"],
        },
        "execution_commands": commands,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "allowed_claim": "paper-facing fresh rerun protocol is frozen, redacted, and dry-run validated",
        "blocked_claims": [
            "completed_720_call_fresh_main_result",
            "fresh_result_without_execute_live_artifact",
            "storage_of_problem_text_reference_answers_prompts_or_api_keys",
        ],
    }


def format_markdown(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    lines = [
        "# Paper Fresh Frozen Rerun Protocol",
        "",
        f"- pass: `{payload['pass']}`",
        f"- target fresh calls: `{m['target_fresh_api_call_count']}`",
        f"- dry-run planned calls: `{m['dry_run_planned_fresh_api_call_count']}`",
        f"- fresh pilot calls: `{m['fresh_pilot_api_call_count']}`",
        f"- available heldout problems: `{m['available_problem_count']}`",
        f"- frozen main problems: `{m['frozen_problem_count']}`",
        f"- protocol ready claim: `{m['fresh_protocol_ready_claim_allowed']}`",
        f"- target result claim: `{m['target_fresh_result_claim_allowed']}`",
        "",
        "## Commands",
        "",
    ]
    for command in payload["execution_commands"]:
        lines.extend([f"### {command['name']}", "", f"```bash\n{command['command']}\n```", ""])
    lines.extend([
        "## Claim Boundary",
        "",
        "This artifact freezes the fresh rerun protocol.  It does not claim the target live rerun has already run.",
    ])
    return "\n".join(lines).rstrip() + "\n"


def _fresh_problem_pool(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows = []
    for path in sorted((root / DEFAULT_PROBLEM_DIR).glob("*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            problem_id = str(row.get("problem_id", ""))
            domain = str(row.get("domain", "unknown"))
            difficulty = str(row.get("difficulty", "unknown"))
            all_rows.append({
                "problem_id": problem_id,
                "domain": domain,
                "difficulty": difficulty,
                "problem_hash": stable_hash([problem_id, domain, difficulty]),
            })
    excluded = _load_existing_problem_ids(root)
    fresh = [row for row in all_rows if row["problem_id"] not in excluded]
    selected = sorted(fresh or all_rows, key=lambda row: (row["domain"], row["difficulty"], row["problem_hash"]))
    domain_counts = dict(Counter(row["domain"] for row in selected))
    report = {
        "problem_dir": str(DEFAULT_PROBLEM_DIR),
        "total_problem_count": len(all_rows),
        "excluded_existing_problem_count": len(excluded),
        "available_problem_count": len(selected),
        "domain_counts": domain_counts,
        "difficulty_counts": dict(Counter(row["difficulty"] for row in selected)),
        "disjoint_from_existing_samples": bool(fresh),
        "problem_pool_hash": stable_hash(selected),
        "sample_problem_rows": selected[:24],
        "fields_retained": ["problem_id", "domain", "difficulty", "problem_hash"],
        "fields_excluded": ["description", "reference_answer", "prompt", "raw_judge_text", "api_secret"],
        "raw_payload_exposed": False,
    }
    return selected, report


def _load_existing_problem_ids(root: Path) -> set[str]:
    out: set[str] = set()
    for rel in DEFAULT_EXISTING_SAMPLES:
        path = root / rel
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, str):
                    out.add(item)
                elif isinstance(item, dict) and item.get("problem_id"):
                    out.add(str(item["problem_id"]))
    return out


def _protocol_manifest(
    *,
    root: Path,
    protocol: dict[str, Any],
    problem_pool: list[dict[str, Any]],
    artifacts: dict[str, dict[str, Any]],
    commands: list[dict[str, str]],
) -> dict[str, Any]:
    source_hashes = {
        name: _sha256(root / SOURCE_ARTIFACTS[name]) if (root / SOURCE_ARTIFACTS[name]).exists() else None
        for name in SOURCE_ARTIFACTS
    }
    code_paths = [
        Path("assumption_os/full_v3_blinded_recursive_live_line.py"),
        Path("assumption_os/paper_frozen_main_experiment_v2.py"),
        Path("assumption_os/paper_fresh_frozen_rerun_protocol.py"),
    ]
    code_hashes = {
        str(path): _sha256(root / path) if (root / path).exists() else None
        for path in code_paths
    }
    return {
        "protocol_hash": stable_hash({
            "protocol": protocol,
            "problem_pool_hash": stable_hash(problem_pool),
            "source_hashes": source_hashes,
            "code_hashes": code_hashes,
            "commands": commands,
        }),
        "source_artifact_hashes": source_hashes,
        "code_hashes": code_hashes,
        "baseline_arms": list(BASELINE_ARMS),
        "full_arm": "full_recursive_morphism_v3",
        "statistical_unit": "problem_id",
        "ci_method": "problem-level bootstrap",
        "paired_test": "problem-level paired utility and sign-test on non-tie outcomes",
        "redaction_policy": {
            "store_problem_text": False,
            "store_reference_answer": False,
            "store_prompt": False,
            "store_judge_text": False,
            "store_api_key": False,
            "retain_problem_id_domain_difficulty_hash": True,
        },
        "completed_target_live_result_claim_allowed": False,
    }


def _commands(*, protocol: dict[str, Any]) -> list[dict[str, str]]:
    seed_arg = ",".join(str(seed) for seed in protocol["seed_values"])
    screen = "phase four/assumption_graph/paper_readiness_20260604/full_v3_blinded_recursive_live_line_20260612.json"
    out = "phase four/assumption_graph/paper_readiness_20260604/paper_fresh_frozen_rerun_live_720_20260612.json"
    execute = (
        "python3 -m assumption_os.full_v3_blinded_recursive_live_line "
        "--execution-mode execute_live "
        f"--generations {protocol['generations']} "
        f"--seeds {seed_arg} "
        f"--candidates-per-generation {protocol['candidates_per_generation']} "
        f"--trigger-rows-per-candidate {protocol['trigger_rows_per_candidate']} "
        f"--control-rows-per-candidate {protocol['control_rows_per_candidate']} "
        f"--model-alias {protocol['model_alias']} "
        f"--parallel-workers {protocol['parallel_workers']} "
        f"--min-planned-calls-for-gate {protocol['target_fresh_api_call_count']} "
        f"--bootstrap-samples {protocol['bootstrap_samples']} "
        f"--screen-artifacts \"{screen}\" "
        f"--eval-id paper_fresh_frozen_rerun_live_720_20260612 "
        f"--out \"{out}\""
    )
    validate = (
        "python3 -m assumption_os.paper_fresh_frozen_rerun_protocol "
        "--eval-id paper_fresh_frozen_rerun_protocol_20260612"
    )
    performance = "python3 -m assumption_os.performance_validation"
    return [
        {
            "name": "validate_fresh_frozen_protocol",
            "command": validate,
            "secret_handling": "No API key required; validates redacted protocol and dry-run plan.",
        },
        {
            "name": "execute_fresh_live_rerun",
            "command": execute,
            "secret_handling": "Requires API credentials in environment/keyfile only; command contains no secret values.",
        },
        {
            "name": "rerun_global_performance_validation",
            "command": performance,
            "secret_handling": "No secret values written to artifact.",
        },
    ]


def _metrics(
    *,
    artifacts: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
    problem_report: dict[str, Any],
    dry_run: dict[str, Any],
    commands: list[dict[str, str]],
    protocol_manifest: dict[str, Any],
) -> dict[str, Any]:
    frozen = artifacts["paper_frozen_main"].get("metrics", {})
    pilot = artifacts["fresh_blinded_pilot"].get("metrics", {})
    leakage = artifacts["simulator_leakage_audit"].get("metrics", {})
    command_text = "\n".join(row["command"] for row in commands)
    return {
        "source_artifact_count": len(SOURCE_ARTIFACTS),
        "source_artifact_pass_rate": round(
            sum(1 for artifact in artifacts.values() if artifact.get("pass")) / len(SOURCE_ARTIFACTS),
            4,
        ),
        "target_fresh_api_call_count": protocol["target_fresh_api_call_count"],
        "seed_count": len(protocol["seed_values"]),
        "generations": protocol["generations"],
        "candidates_per_generation": protocol["candidates_per_generation"],
        "bootstrap_samples": protocol["bootstrap_samples"],
        "available_problem_count": problem_report["available_problem_count"],
        "available_problem_domain_count": len(problem_report["domain_counts"]),
        "excluded_existing_problem_count": problem_report["excluded_existing_problem_count"],
        "disjoint_from_existing_samples": problem_report["disjoint_from_existing_samples"],
        "problem_manifest_raw_payload_exposed": problem_report["raw_payload_exposed"],
        "problem_pool_hash": problem_report["problem_pool_hash"],
        "frozen_problem_count": int(frozen.get("problem_count") or 0),
        "baseline_count": int(frozen.get("baseline_count") or len(BASELINE_ARMS)),
        "frozen_margin_over_best_baseline": float(frozen.get("full_v3_margin_over_best_baseline_score") or 0.0),
        "frozen_min_pairwise_utility": float(frozen.get("min_pairwise_utility") or 0.0),
        "fresh_pilot_pass": bool(artifacts["fresh_blinded_pilot"].get("pass")),
        "fresh_pilot_api_call_count": int(pilot.get("fresh_api_call_count") or 0),
        "fresh_pilot_planned_call_count": int(pilot.get("planned_fresh_api_call_count") or 0),
        "fresh_pilot_live_error_count": int(pilot.get("live_error_count") or 0),
        "fresh_pilot_accepted_count": int(pilot.get("accepted_count") or 0),
        "fresh_pilot_rejected_count": int(pilot.get("rejected_count") or 0),
        "fresh_pilot_prompt_answer_or_secret_payload_detected": bool(
            pilot.get("prompt_answer_or_secret_payload_detected")
        ),
        "dry_run_pass": bool(dry_run.get("pass")),
        "dry_run_planned_fresh_api_call_count": int(dry_run["metrics"]["planned_fresh_api_call_count"]),
        "dry_run_selected_candidate_count": int(dry_run["metrics"]["selected_candidate_count"]),
        "dry_run_trigger_problem_count": int(dry_run["metrics"]["trigger_problem_count"]),
        "dry_run_control_problem_count": int(dry_run["metrics"]["control_problem_count"]),
        "dry_run_prompt_answer_or_secret_payload_detected": bool(
            dry_run["metrics"]["prompt_answer_or_secret_payload_detected"]
        ),
        "simulator_leakage_audit_pass": bool(artifacts["simulator_leakage_audit"].get("pass")),
        "simulator_state_feature_leak_count": int(leakage.get("state_feature_leak_count", 0)),
        "simulator_prediction_identity_count": int(leakage.get("prediction_outcome_exact_identity_count", 0)),
        "source_artifact_hash_count": sum(
            1 for value in protocol_manifest["source_artifact_hashes"].values() if value
        ),
        "code_hash_count": sum(1 for value in protocol_manifest["code_hashes"].values() if value),
        "command_count": len(commands),
        "command_secret_hit_count": len(SECRET_RE.findall(command_text)),
        "fresh_protocol_ready_claim_allowed": True,
        "target_fresh_result_claim_allowed": False,
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper fresh frozen rerun protocol artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="paper_fresh_frozen_rerun_protocol_20260612")
    parser.add_argument("--target-fresh-api-call-count", type=int, default=DEFAULT_PROTOCOL["target_fresh_api_call_count"])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_paper_fresh_frozen_rerun_protocol_payload(
        root=root,
        eval_id=args.eval_id,
        target_fresh_api_call_count=args.target_fresh_api_call_count,
    )
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out:
        md_out = Path(args.md_out)
        md_out = md_out if md_out.is_absolute() else root / md_out
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
