"""Retry HLE endpoint-error manifest rows with clean replacement semantics.

This helper is intentionally narrow: it only reruns rows listed in an
``endpoint_retry_manifest`` and then delegates replacement validation to
``hle_parallel_shard_runner --combine-split-runs``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = Path("phase four/assumption_graph/paper_readiness_20260604/hle_parallel_runs")
DEFAULT_MD_DIR = Path("reconstruction/md")


@dataclass(frozen=True)
class RetryGroup:
    model: str
    variant: str
    seed_offsets: tuple[int, ...]
    retry_keys: tuple[str, ...]


def load_retry_groups(source_run_json: Path) -> list[RetryGroup]:
    payload = json.loads(source_run_json.read_text(encoding="utf-8"))
    manifest = payload.get("endpoint_retry_manifest") if isinstance(payload, dict) else {}
    items = manifest.get("retry_items") if isinstance(manifest, dict) else []
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        model = str(item.get("model") or "").strip()
        variant = str(item.get("variant") or "").strip()
        retry_key = str(item.get("retry_key") or "").strip()
        try:
            seed_offset = int(item.get("seed_offset"))
        except (TypeError, ValueError):
            continue
        if not model or not variant or not retry_key:
            continue
        bucket = grouped.setdefault((model, variant), {"seed_offsets": set(), "retry_keys": []})
        bucket["seed_offsets"].add(seed_offset)
        bucket["retry_keys"].append(retry_key)
    groups = [
        RetryGroup(
            model=model,
            variant=variant,
            seed_offsets=tuple(sorted(bucket["seed_offsets"])),
            retry_keys=tuple(sorted(dict.fromkeys(bucket["retry_keys"]))),
        )
        for (model, variant), bucket in sorted(grouped.items())
    ]
    return groups


def build_retry_command(
    *,
    group: RetryGroup,
    eval_id: str,
    run_dir: Path,
    md_dir: Path,
    parallel_workers: int,
    model_router_attempts: int,
    model_router_transient_extra_attempts: int,
    model_router_per_attempt_timeout: float,
    model_router_reasoning_effort: str,
    max_tokens: int,
    model_router_no_byte_timeout_sec: float,
    model_router_global_concurrency: int,
    live_model_preflight_probe_count: int,
    live_model_preflight_max_error_rate: float,
    live_model_preflight_timeout_sec: float,
    live_model_preflight_prompt_chars: int,
    live_model_preflight_max_tokens: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "assumption_os.hle_parallel_shard_runner",
        "--eval-id",
        eval_id,
        "--total-sample-size",
        str(len(group.seed_offsets)),
        "--shard-size",
        "1",
        "--parallel-workers",
        str(parallel_workers),
        "--seed-offsets",
        ",".join(str(seed) for seed in group.seed_offsets),
        "--generalization-holdout",
        "--generalization-holdout-preserve-explicit-seed-offsets",
        "--sample-answer-type",
        "multipleChoice",
        "--models",
        group.model,
        "--variants",
        group.variant,
        "--execute-live",
        "--max-tokens",
        str(max_tokens),
        "--model-router-attempts",
        str(model_router_attempts),
        "--model-router-transient-extra-attempts",
        str(model_router_transient_extra_attempts),
        "--model-router-per-attempt-timeout",
        str(model_router_per_attempt_timeout),
        "--model-router-no-byte-timeout-sec",
        str(model_router_no_byte_timeout_sec),
        "--model-router-subprocess-calls",
        "--model-router-global-concurrency",
        str(model_router_global_concurrency),
        "--model-router-global-slot-ttl-sec",
        "900",
        "--model-router-global-slot-wait-sec",
        "600",
        "--live-model-preflight-probe-count",
        str(live_model_preflight_probe_count),
        "--live-model-preflight-max-error-rate",
        str(live_model_preflight_max_error_rate),
        "--live-model-preflight-timeout-sec",
        str(live_model_preflight_timeout_sec),
        "--live-model-preflight-prompt-chars",
        str(live_model_preflight_prompt_chars),
        "--live-model-preflight-max-tokens",
        str(live_model_preflight_max_tokens),
        "--run-dir",
        str(run_dir),
        "--md-dir",
        str(md_dir),
    ]
    if str(model_router_reasoning_effort or "").strip():
        command.extend(
            ["--model-router-reasoning-effort", str(model_router_reasoning_effort).strip()]
        )
    return command


def build_combine_command(
    *,
    eval_id: str,
    source_run_json: Path,
    retry_outputs: list[Path],
    run_dir: Path,
    md_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "assumption_os.hle_parallel_shard_runner",
        "--eval-id",
        eval_id,
        "--combine-split-runs",
        "--split-run-input",
        str(source_run_json),
        "--split-retry-inputs",
        ",".join(str(path) for path in retry_outputs),
        "--allow-split-retry-clean-replacements",
        "--run-dir",
        str(run_dir),
        "--md-dir",
        str(md_dir),
    ]


def _cache_only_env() -> dict[str, str]:
    return {
        "HLE_EVIDENCE_SOURCE_CACHE_ONLY": "1",
        "HLE_SOURCE_SEARCH_CACHE_ONLY": "1",
        "HLE_LIVE_SOURCE_SEARCH_DISABLED": "1",
        "HLE_LIVE_SOURCE_SEARCH_ALLOWED": "0",
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _retry_success(path: Path) -> bool:
    payload = _load_json(path)
    errors = payload.get("error_stratification") if isinstance(payload, dict) else {}
    return int((errors or {}).get("top_level_error_count") or 0) == 0


def run_retry_manifest(args: argparse.Namespace) -> dict[str, Any]:
    source_run_json = Path(args.source_run_json)
    run_dir = Path(args.run_dir)
    md_dir = Path(args.md_dir)
    groups = load_retry_groups(source_run_json)
    if not groups:
        return {
            "pass": True,
            "reason": "no_retry_items",
            "source_run_json": str(source_run_json),
            "retry_group_count": 0,
            "raw_content_persisted": False,
        }
    retry_outputs: list[Path] = []
    group_results: list[dict[str, Any]] = []
    env = {**os.environ, **_cache_only_env()}
    for group_index, group in enumerate(groups):
        success_path: Path | None = None
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, int(args.max_wait_attempts) + 1):
            eval_id = f"{args.eval_id_prefix}_retry_g{group_index:02d}_a{attempt:02d}_{group.variant}"
            command = build_retry_command(
                group=group,
                eval_id=eval_id,
                run_dir=run_dir,
                md_dir=md_dir,
                parallel_workers=args.parallel_workers,
                model_router_attempts=args.model_router_attempts,
                model_router_transient_extra_attempts=args.model_router_transient_extra_attempts,
                model_router_per_attempt_timeout=args.model_router_per_attempt_timeout,
                model_router_reasoning_effort=args.model_router_reasoning_effort,
                max_tokens=args.max_tokens,
                model_router_no_byte_timeout_sec=args.model_router_no_byte_timeout_sec,
                model_router_global_concurrency=args.model_router_global_concurrency,
                live_model_preflight_probe_count=args.live_model_preflight_probe_count,
                live_model_preflight_max_error_rate=args.live_model_preflight_max_error_rate,
                live_model_preflight_timeout_sec=args.live_model_preflight_timeout_sec,
                live_model_preflight_prompt_chars=args.live_model_preflight_prompt_chars,
                live_model_preflight_max_tokens=args.live_model_preflight_max_tokens,
            )
            out_path = run_dir / f"{eval_id}.json"
            attempts.append({
                "attempt": attempt,
                "eval_id": eval_id,
                "out": str(out_path),
                "command": command if args.dry_run else None,
            })
            if args.dry_run:
                success_path = out_path
                break
            subprocess.run(command, check=False, env={**dict(), **env})
            if _retry_success(out_path):
                success_path = out_path
                break
            if attempt < int(args.max_wait_attempts):
                time.sleep(float(args.sleep_sec))
        retry_outputs.extend([success_path] if success_path is not None else [])
        group_results.append({
            "model": group.model,
            "variant": group.variant,
            "seed_offsets": list(group.seed_offsets),
            "retry_keys": list(group.retry_keys),
            "success": success_path is not None and (args.dry_run or _retry_success(success_path)),
            "success_out": str(success_path) if success_path else None,
            "attempts": attempts,
        })
    combine_out: Path | None = None
    combine_payload: dict[str, Any] = {}
    if retry_outputs and not args.dry_run:
        combine_eval_id = f"{args.eval_id_prefix}_combined"
        combine_command = build_combine_command(
            eval_id=combine_eval_id,
            source_run_json=source_run_json,
            retry_outputs=retry_outputs,
            run_dir=run_dir,
            md_dir=md_dir,
        )
        subprocess.run(combine_command, check=False, env=env)
        combine_out = run_dir / f"{combine_eval_id}.json"
        combine_payload = _load_json(combine_out)
    return {
        "pass": bool(combine_payload.get("paper_clean_pass")) if combine_payload else all(
            item["success"] for item in group_results
        ),
        "source_run_json": str(source_run_json),
        "retry_group_count": len(groups),
        "retry_outputs": [str(path) for path in retry_outputs],
        "combined_out": str(combine_out) if combine_out else None,
        "combined_paper_clean_pass": combine_payload.get("paper_clean_pass"),
        "combined_paper_clean_failed_gates": combine_payload.get("paper_clean_failed_gates"),
        "groups": group_results,
        "raw_content_persisted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-json", required=True)
    parser.add_argument("--eval-id-prefix", required=True)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--md-dir", default=str(DEFAULT_MD_DIR))
    parser.add_argument("--max-wait-attempts", type=int, default=1)
    parser.add_argument("--sleep-sec", type=float, default=300.0)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--model-router-attempts", type=int, default=2)
    parser.add_argument("--model-router-transient-extra-attempts", type=int, default=0)
    parser.add_argument("--model-router-per-attempt-timeout", type=float, default=0.0)
    parser.add_argument("--model-router-reasoning-effort", default="")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--model-router-no-byte-timeout-sec", type=float, default=600.0)
    parser.add_argument("--model-router-global-concurrency", type=int, default=1)
    parser.add_argument("--live-model-preflight-probe-count", type=int, default=1)
    parser.add_argument("--live-model-preflight-max-error-rate", type=float, default=0.0)
    parser.add_argument("--live-model-preflight-timeout-sec", type=float, default=60.0)
    parser.add_argument("--live-model-preflight-prompt-chars", type=int, default=12000)
    parser.add_argument("--live-model-preflight-max-tokens", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_retry_manifest(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
