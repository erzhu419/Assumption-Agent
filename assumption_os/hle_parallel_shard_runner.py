"""Parallel shard runner for HLE smoke evaluations.

This module orchestrates multiple ``hle_smoke_eval`` subprocesses.  It does
not change the underlying scoring path; it only adds bounded parallelism,
heartbeat files, optional soft-timeout observation, and error-stratified
aggregate reports.
Artifacts intentionally store hashes, counts, and metadata only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR
from .diagnostic_logging import JsonlDiagnosticLogger, log_event
from .hle_smoke_eval import (
    DATASET_NAME,
    HLE_OFFICIAL_SOURCES,
    _aggregate_rows,
    _agent_meets_best_control_gate,
    _component_efficacy_summary,
    _collect_existing_hle_problem_hashes,
    _control_comparison,
    _expected_but_missing_modules,
    _has_image_payload,
    _load_hle_test_dataset,
    _load_text_only_sample,
    _module_activation_summary,
    _operator_activation_summary,
    _operator_application_summary,
    _problem_from_row,
    _route_credit_table,
)


DEFAULT_RUN_DIR = PAPER_DIR / "hle_parallel_runs"
DEFAULT_MD_DIR = Path("reconstruction/md")
DEFAULT_HLE_DATASET_LOCAL_PATH = PAPER_DIR / "hle_dataset_cache" / "test"
DEFAULT_HLE_EVIDENCE_SOURCE_CACHE_DIR = PAPER_DIR / "hle_evidence_source_cache"
ERROR_EVENT_NAMES = {
    "call_error",
    "recursive_child_error",
    "recursive_child_timeout",
    "candidate_claim_verifier_error",
    "counter_assumption_verifier_error",
    "source_grounded_mc_verifier_error",
    "option_evidence_arbitrator_error",
    "domain_rule_mc_verifier_error",
    "critic_synthesis_child_error",
    "math_tool_child_error",
}
PAPER_CLEAN_STANDARD_CONTROL_VARIANTS = ("raw", "hipporag_baseline")
PAPER_CLEAN_BUDGET_MATCHED_CONTROL_VARIANTS = ("raw_budget_matched", "hipporag_budget_matched")


@dataclass(frozen=True)
class ShardSpec:
    shard_index: int
    eval_id: str
    sample_size: int
    seed_offset: int
    out: Path
    md_out: Path
    log_out: Path
    stdout_out: Path


@dataclass
class ShardRunState:
    spec: ShardSpec
    command: list[str]
    process: subprocess.Popen[Any] | None = None
    started_monotonic: float | None = None
    finished_monotonic: float | None = None
    returncode: int | None = None
    status: str = "pending"
    soft_timeout_sent: bool = False
    soft_timeout_observed: bool = False
    hard_kill_sent: bool = False
    reused_existing_payload: bool = False
    process_timeout_policy: str = "watch_only"
    error: str | None = None
    last_process_memory: dict[str, Any] = field(default_factory=dict)
    peak_rss_kb: int | None = None
    peak_vms_kb: int | None = None
    _stdout_handle: Any = field(default=None, repr=False, compare=False)

    def elapsed_sec(self, now: float | None = None) -> float | None:
        if self.started_monotonic is None:
            return None
        end = self.finished_monotonic if self.finished_monotonic is not None else (now or time.monotonic())
        return round(max(0.0, end - self.started_monotonic), 4)


def build_shard_specs(
    *,
    eval_id: str,
    total_sample_size: int,
    shard_size: int,
    seed_offset: int,
    seed_stride: int,
    run_dir: Path,
    md_dir: Path,
) -> list[ShardSpec]:
    if total_sample_size <= 0:
        raise ValueError("total_sample_size must be positive")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    shard_count = math.ceil(total_sample_size / shard_size)
    specs: list[ShardSpec] = []
    for shard_index in range(shard_count):
        current_size = min(shard_size, total_sample_size - shard_index * shard_size)
        shard_eval_id = f"{eval_id}_shard_{shard_index:03d}"
        specs.append(
            ShardSpec(
                shard_index=shard_index,
                eval_id=shard_eval_id,
                sample_size=current_size,
                seed_offset=seed_offset + shard_index * seed_stride,
                out=run_dir / f"{shard_eval_id}.json",
                md_out=md_dir / f"{shard_eval_id}.md",
                log_out=run_dir / f"{shard_eval_id}.jsonl",
                stdout_out=run_dir / f"{shard_eval_id}.stdout.log",
            )
        )
    return specs


def build_shard_specs_for_seed_offsets(
    *,
    eval_id: str,
    seed_offsets: list[int],
    run_dir: Path,
    md_dir: Path,
) -> list[ShardSpec]:
    if not seed_offsets:
        raise ValueError("seed_offsets must not be empty")
    specs: list[ShardSpec] = []
    for shard_index, seed_offset in enumerate(seed_offsets):
        shard_eval_id = f"{eval_id}_shard_{shard_index:03d}"
        specs.append(
            ShardSpec(
                shard_index=shard_index,
                eval_id=shard_eval_id,
                sample_size=1,
                seed_offset=int(seed_offset),
                out=run_dir / f"{shard_eval_id}.json",
                md_out=md_dir / f"{shard_eval_id}.md",
                log_out=run_dir / f"{shard_eval_id}.jsonl",
                stdout_out=run_dir / f"{shard_eval_id}.stdout.log",
            )
        )
    return specs


def _parse_seed_offsets(value: str | None) -> list[int]:
    if not value:
        return []
    offsets: list[int] = []
    for item in str(value).split(","):
        stripped = item.strip()
        if not stripped:
            continue
        offsets.append(int(stripped))
    return offsets


def dedupe_shard_specs_by_sample_hash(
    *,
    root: Path,
    specs: list[ShardSpec],
    max_scan: int,
    seed_stride: int,
    exclude_existing_hle_artifacts: bool,
    exclude_artifact_glob: str,
    sample_answer_type: str,
    sample_subject_contains: str,
    max_attempts: int = 25,
    sample_loader: Any | None = None,
) -> tuple[list[ShardSpec], dict[str, Any]]:
    """Advance shard seeds until parent-side sample hashes are non-overlapping.

    The preflight reads HLE rows only in memory through the same text-only loader
    used by child shards.  It persists only hashes and seed remaps.
    """
    if not specs:
        return specs, {"enabled": True, "status": "empty", "raw_content_persisted": False, "remaps": []}
    loader = sample_loader or _load_text_only_sample
    excluded_hashes = (
        _collect_existing_hle_problem_hashes(root=root, artifact_glob=exclude_artifact_glob)
        if exclude_existing_hle_artifacts
        else set()
    )
    if sample_loader is None and all(spec.sample_size == 1 for spec in specs):
        try:
            return _dedupe_single_row_shards_by_scan_index(
                specs=specs,
                max_scan=max_scan,
                seed_stride=seed_stride,
                exclude_problem_hashes=excluded_hashes,
                answer_type_filter=sample_answer_type,
                subject_contains=sample_subject_contains,
                max_attempts=max_attempts,
            )
        except Exception as exc:
            fallback_specs, fallback_summary = _dedupe_shard_specs_by_sample_hash_slow(
                specs=specs,
                loader=loader,
                max_scan=max_scan,
                seed_stride=seed_stride,
                exclude_problem_hashes=excluded_hashes,
                sample_answer_type=sample_answer_type,
                sample_subject_contains=sample_subject_contains,
                max_attempts=max_attempts,
            )
            fallback_summary.update({
                "fast_single_pass": False,
                "fast_single_pass_error_type": type(exc).__name__,
                "fast_single_pass_error": str(exc)[:240],
            })
            return fallback_specs, fallback_summary
    return _dedupe_shard_specs_by_sample_hash_slow(
        specs=specs,
        loader=loader,
        max_scan=max_scan,
        seed_stride=seed_stride,
        exclude_problem_hashes=excluded_hashes,
        sample_answer_type=sample_answer_type,
        sample_subject_contains=sample_subject_contains,
        max_attempts=max_attempts,
    )


def _load_text_only_candidate_index(
    *,
    max_raw_scan: int,
    exclude_problem_hashes: set[str],
    answer_type_filter: str,
    subject_contains: str,
) -> list[dict[str, Any]]:
    dataset = _load_hle_test_dataset()
    candidates: list[dict[str, Any]] = []
    skipped = 0
    for scanned, row in enumerate(dataset, start=1):
        if scanned > max_raw_scan:
            break
        if _has_image_payload(row):
            skipped += 1
            continue
        if not str(row.get("question") or "").strip() or not str(row.get("answer") or "").strip():
            skipped += 1
            continue
        if answer_type_filter and str(row.get("answer_type") or "") != answer_type_filter:
            skipped += 1
            continue
        if subject_contains:
            haystack = " ".join([
                str(row.get("category") or ""),
                str(row.get("raw_subject") or ""),
            ]).lower()
            if subject_contains.lower() not in haystack:
                skipped += 1
                continue
        problem = _problem_from_row(row, scanned=scanned, skipped_before=skipped)
        problem_hash = str(problem.get("id_hash") or "")
        if not problem_hash or problem_hash in exclude_problem_hashes:
            skipped += 1
            continue
        candidates.append({
            "scanned_index": scanned,
            "id_hash": problem_hash,
        })
    return candidates


def _dedupe_single_row_shards_by_scan_index(
    *,
    specs: list[ShardSpec],
    max_scan: int,
    seed_stride: int,
    exclude_problem_hashes: set[str],
    answer_type_filter: str,
    subject_contains: str,
    max_attempts: int,
) -> tuple[list[ShardSpec], dict[str, Any]]:
    """Fast dedupe for sample_size=1 shards using one HF streaming pass.

    The child loader chooses the first valid text-only row with scanned_index
    greater than seed_offset.  This preflight therefore selects seed offsets
    that reproduce a distinct first valid row in each child shard without
    persisting raw question or answer text.
    """
    if not specs:
        return specs, {"enabled": True, "status": "empty", "raw_content_persisted": False, "remaps": []}
    stride = max(1, seed_stride)
    max_attempt_count = max(1, max_attempts + 1)
    max_original_seed = max(int(spec.seed_offset) for spec in specs)
    max_raw_scan = max_original_seed + max(1, max_scan) + max_attempt_count * stride + 2
    candidates = _load_text_only_candidate_index(
        max_raw_scan=max_raw_scan,
        exclude_problem_hashes=exclude_problem_hashes,
        answer_type_filter=answer_type_filter,
        subject_contains=subject_contains,
    )
    positions = [int(row["scanned_index"]) for row in candidates]
    seen: set[str] = set()
    deduped: list[ShardSpec] = []
    remaps: list[dict[str, Any]] = []
    for spec in specs:
        original_seed = int(spec.seed_offset)
        candidate_seed = original_seed
        selected_spec = spec
        selected_hashes: list[str] = []
        selected_status = "fallback_unchecked"
        duplicate_hashes: list[str] = []
        attempt_index = 0
        while attempt_index < max_attempt_count:
            upper_scan = candidate_seed + max(1, max_scan)
            candidate_index = bisect_right(positions, candidate_seed)
            wrapped_candidate = False
            if candidate_index >= len(candidates) or positions[candidate_index] > upper_scan:
                if not candidates:
                    selected_status = "insufficient_sample"
                    selected_hashes = []
                    candidate_seed += stride
                    attempt_index += 1
                    continue
                candidate_index = (candidate_seed + attempt_index) % len(candidates)
                wrapped_candidate = True
            candidate = candidates[candidate_index]
            candidate_hash = str(candidate.get("id_hash") or "")
            if candidate_hash and candidate_hash not in seen:
                selected_seed = max(0, int(candidate.get("scanned_index") or 1) - 1)
                selected_spec = replace(spec, seed_offset=selected_seed)
                selected_hashes = [candidate_hash]
                selected_status = "accepted_wrapped" if wrapped_candidate else "accepted"
                duplicate_hashes = []
                break
            duplicate_hashes = [candidate_hash] if candidate_hash else []
            selected_hashes = [candidate_hash] if candidate_hash else []
            selected_status = "duplicate"
            candidate_seed = max(candidate_seed + stride, int(candidate.get("scanned_index") or candidate_seed))
            attempt_index += 1
        deduped.append(selected_spec)
        seen.update(selected_hashes)
        remaps.append({
            "shard_index": spec.shard_index,
            "original_seed_offset": original_seed,
            "selected_seed_offset": selected_spec.seed_offset,
            "status": selected_status,
            "attempt_count": attempt_index + 1,
            "duplicate_hashes": duplicate_hashes,
            "selected_problem_hashes": selected_hashes,
        })
    return deduped, {
        "enabled": True,
        "status": "ok",
        "fast_single_pass": True,
        "candidate_hash_count": len(candidates),
        "candidate_scan_limit": max_raw_scan,
        "excluded_hash_count": len(exclude_problem_hashes),
        "raw_content_persisted": False,
        "accepted_shard_count": sum(1 for row in remaps if str(row.get("status") or "").startswith("accepted")),
        "duplicate_fallback_count": sum(1 for row in remaps if row.get("status") == "duplicate"),
        "distinct_problem_hash_count": len(seen),
        "remaps": remaps,
    }


def _dedupe_shard_specs_by_sample_hash_slow(
    *,
    specs: list[ShardSpec],
    loader: Any,
    max_scan: int,
    seed_stride: int,
    exclude_problem_hashes: set[str],
    sample_answer_type: str,
    sample_subject_contains: str,
    max_attempts: int,
) -> tuple[list[ShardSpec], dict[str, Any]]:
    if not specs:
        return specs, {"enabled": True, "status": "empty", "raw_content_persisted": False, "remaps": []}
    seen: set[str] = set()
    deduped: list[ShardSpec] = []
    remaps: list[dict[str, Any]] = []
    for spec in specs:
        original_seed = spec.seed_offset
        candidate_seed = original_seed
        selected_spec = spec
        selected_hashes: list[str] = []
        selected_status = "fallback_unchecked"
        for attempt_index in range(max(1, max_attempts + 1)):
            rows = loader(
                sample_size=spec.sample_size,
                max_scan=max_scan + max(0, candidate_seed),
                seed_offset=candidate_seed,
                exclude_problem_hashes=exclude_problem_hashes,
                answer_type_filter=sample_answer_type,
                subject_contains=sample_subject_contains,
            )
            hashes = [str(row.get("id_hash")) for row in rows if row.get("id_hash")]
            duplicate_hashes = sorted(set(hashes) & seen)
            enough_rows = len(hashes) >= spec.sample_size
            if enough_rows and not duplicate_hashes:
                selected_spec = replace(spec, seed_offset=candidate_seed)
                selected_hashes = hashes
                selected_status = "accepted"
                break
            selected_hashes = hashes
            selected_status = "duplicate" if duplicate_hashes else "insufficient_sample"
            candidate_seed += max(1, seed_stride)
        deduped.append(selected_spec)
        seen.update(selected_hashes)
        remaps.append(
            {
                "shard_index": spec.shard_index,
                "original_seed_offset": original_seed,
                "selected_seed_offset": selected_spec.seed_offset,
                "status": selected_status,
                "attempt_count": attempt_index + 1,
                "sample_count": len(selected_hashes),
                "sample_problem_hashes": selected_hashes,
                "raw_content_persisted": False,
            }
        )
    duplicate_count = sum(1 for row in remaps if row["status"] == "duplicate")
    return deduped, {
        "enabled": True,
        "status": "completed",
        "max_attempts": max_attempts,
        "seed_stride": seed_stride,
        "excluded_existing_problem_count": len(exclude_problem_hashes),
        "accepted_shard_count": sum(1 for row in remaps if row["status"] == "accepted"),
        "duplicate_fallback_count": duplicate_count,
        "remaps": remaps,
        "raw_content_persisted": False,
    }


def build_shard_command(
    spec: ShardSpec,
    *,
    root: Path,
    max_scan: int,
    models: str,
    variants: str,
    execute_live: bool,
    call_timeout: float | None,
    max_tokens: int,
    graph_dir: Path,
    agent_top_k: int,
    agent_context_max_chars: int,
    agent_child_mode: str,
    agent_child_timeout: float | None,
    evidence_bridge_enabled: bool,
    exclude_existing_hle_artifacts: bool,
    exclude_artifact_glob: str,
    sample_answer_type: str,
    sample_subject_contains: str,
    enable_assumption_operators: bool = False,
    disable_assumption_operators: bool = False,
    assumption_operator_domains: str = "",
    assumption_operator_skip_domains: str = "",
    assumption_operator_max_specs: int | None = None,
    allow_assumption_operators_without_context: bool = False,
    enable_assumption_operator_retrieval_fallback: bool = False,
    assumption_operator_fallback_min_score: float | None = None,
    enable_operator_application_verifier: bool = False,
    disable_domain_rule_verifier: bool = False,
    enable_option_claim_contrastive_adjudicator: bool = False,
    disable_option_claim_contrastive_adjudicator: bool = False,
    enable_option_claim_span_directness_verifier: bool = False,
    disable_option_claim_span_directness_verifier: bool = False,
) -> list[str]:
    effective_max_scan = max_scan + max(0, spec.seed_offset)
    cmd = [
        sys.executable,
        "-m",
        "assumption_os.hle_smoke_eval",
        "--root",
        str(root),
        "--eval-id",
        spec.eval_id,
        "--sample-size",
        str(spec.sample_size),
        "--max-scan",
        str(effective_max_scan),
        "--seed-offset",
        str(spec.seed_offset),
        "--models",
        models,
        "--variants",
        variants,
        "--max-tokens",
        str(max_tokens),
        "--log-out",
        str(spec.log_out),
        "--graph-dir",
        str(graph_dir),
        "--agent-top-k",
        str(agent_top_k),
        "--agent-context-max-chars",
        str(agent_context_max_chars),
        "--agent-child-mode",
        agent_child_mode,
        "--out",
        str(spec.out),
        "--md-out",
        str(spec.md_out),
        "--hard-exit-after-write",
    ]
    if execute_live:
        cmd.append("--execute-live")
    if call_timeout is not None:
        cmd.extend(["--call-timeout", str(call_timeout)])
    if agent_child_timeout is not None:
        cmd.extend(["--agent-child-timeout", str(agent_child_timeout)])
    if not evidence_bridge_enabled:
        cmd.append("--disable-evidence-bridge")
    if enable_assumption_operators:
        cmd.append("--enable-assumption-operators")
    if disable_assumption_operators:
        cmd.append("--disable-assumption-operators")
    if assumption_operator_domains:
        cmd.extend(["--assumption-operator-domains", assumption_operator_domains])
    if assumption_operator_skip_domains:
        cmd.extend(["--assumption-operator-skip-domains", assumption_operator_skip_domains])
    if assumption_operator_max_specs is not None:
        cmd.extend(["--assumption-operator-max-specs", str(assumption_operator_max_specs)])
    if allow_assumption_operators_without_context:
        cmd.append("--allow-assumption-operators-without-context")
    if enable_assumption_operator_retrieval_fallback:
        cmd.append("--enable-assumption-operator-retrieval-fallback")
    if assumption_operator_fallback_min_score is not None:
        cmd.extend([
            "--assumption-operator-fallback-min-score",
            str(assumption_operator_fallback_min_score),
        ])
    if enable_operator_application_verifier:
        cmd.append("--enable-operator-application-verifier")
    if disable_domain_rule_verifier:
        cmd.append("--disable-domain-rule-verifier")
    if enable_option_claim_contrastive_adjudicator:
        cmd.append("--enable-option-claim-contrastive-adjudicator")
    if disable_option_claim_contrastive_adjudicator:
        cmd.append("--disable-option-claim-contrastive-adjudicator")
    if enable_option_claim_span_directness_verifier:
        cmd.append("--enable-option-claim-span-directness-verifier")
    if disable_option_claim_span_directness_verifier:
        cmd.append("--disable-option-claim-span-directness-verifier")
    if exclude_existing_hle_artifacts:
        cmd.append("--exclude-existing-hle-artifacts")
    if exclude_artifact_glob:
        cmd.extend(["--exclude-artifact-glob", exclude_artifact_glob])
    if sample_answer_type:
        cmd.extend(["--sample-answer-type", sample_answer_type])
    if sample_subject_contains:
        cmd.extend(["--sample-subject-contains", sample_subject_contains])
    return cmd


def build_runner_env(
    *,
    model_router_attempts: int | None,
    model_router_timeout: float | None,
    parallel_workers: int | None = None,
    model_router_per_attempt_timeout: float | None = None,
    model_router_subprocess_calls: bool | None = None,
    model_router_no_byte_timeout_sec: float | None = None,
    model_router_backoff_base_sec: float | None = None,
    model_router_global_concurrency: int | None = None,
    model_router_global_concurrency_dir: str | None = None,
    model_router_global_slot_ttl_sec: float | None = None,
    model_router_global_slot_wait_sec: float | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    apply_hle_offline_defaults(env)
    if parallel_workers is not None:
        env["HLE_PARALLEL_SHARD_WORKERS"] = str(max(1, int(parallel_workers)))
    if model_router_attempts is not None:
        env["MODEL_ROUTER_ATTEMPTS"] = str(model_router_attempts)
    if model_router_timeout is not None:
        env["MODEL_ROUTER_TIMEOUT"] = str(model_router_timeout)
    if model_router_per_attempt_timeout is not None:
        env["MODEL_ROUTER_PER_ATTEMPT_TIMEOUT"] = str(model_router_per_attempt_timeout)
    if model_router_subprocess_calls is not None:
        env["MODEL_ROUTER_SUBPROCESS_CALLS"] = "1" if model_router_subprocess_calls else "0"
    if model_router_no_byte_timeout_sec is not None:
        env["MODEL_ROUTER_SUBPROCESS_NO_BYTE_TIMEOUT_SEC"] = str(model_router_no_byte_timeout_sec)
    if model_router_backoff_base_sec is not None:
        env["MODEL_ROUTER_BACKOFF_BASE_SEC"] = str(model_router_backoff_base_sec)
    if model_router_global_concurrency is not None:
        env["MODEL_ROUTER_GLOBAL_CONCURRENCY"] = str(model_router_global_concurrency)
    if model_router_global_concurrency_dir:
        env["MODEL_ROUTER_GLOBAL_CONCURRENCY_DIR"] = model_router_global_concurrency_dir
    if model_router_global_slot_ttl_sec is not None:
        env["MODEL_ROUTER_GLOBAL_SLOT_TTL_SEC"] = str(model_router_global_slot_ttl_sec)
    if model_router_global_slot_wait_sec is not None:
        env["MODEL_ROUTER_GLOBAL_SLOT_WAIT_SEC"] = str(model_router_global_slot_wait_sec)
    if os.environ.get("HLE_EXISTING_HASH_CACHE_PATH"):
        env["HLE_EXISTING_HASH_CACHE_PATH"] = os.environ["HLE_EXISTING_HASH_CACHE_PATH"]
    if os.environ.get("HLE_EXISTING_HASH_CACHE_ALLOW_STALE"):
        env["HLE_EXISTING_HASH_CACHE_ALLOW_STALE"] = os.environ["HLE_EXISTING_HASH_CACHE_ALLOW_STALE"]
    return env


def model_router_policy_from_env(env: dict[str, str]) -> dict[str, Any]:
    return {
        "attempts": env.get("MODEL_ROUTER_ATTEMPTS"),
        "timeout_sec": env.get("MODEL_ROUTER_TIMEOUT"),
        "per_attempt_timeout_sec": env.get("MODEL_ROUTER_PER_ATTEMPT_TIMEOUT"),
        "subprocess_calls": env.get("MODEL_ROUTER_SUBPROCESS_CALLS"),
        "subprocess_no_byte_timeout_sec": env.get("MODEL_ROUTER_SUBPROCESS_NO_BYTE_TIMEOUT_SEC")
        or env.get("MODEL_ROUTER_NO_BYTE_TIMEOUT_SEC"),
        "backoff_base_sec": env.get("MODEL_ROUTER_BACKOFF_BASE_SEC"),
        "global_concurrency": env.get("MODEL_ROUTER_GLOBAL_CONCURRENCY"),
        "global_slot_ttl_sec": env.get("MODEL_ROUTER_GLOBAL_SLOT_TTL_SEC"),
        "global_slot_wait_sec": env.get("MODEL_ROUTER_GLOBAL_SLOT_WAIT_SEC"),
        "parallel_shard_workers": env.get("HLE_PARALLEL_SHARD_WORKERS"),
        "router_aware_child_worker_cap_enabled": env.get("HLE_ENABLE_ROUTER_AWARE_CHILD_WORKER_CAP"),
        "router_aware_child_workers_per_shard": env.get("HLE_ROUTER_AWARE_CHILD_WORKERS_PER_SHARD"),
        "router_aware_child_worker_cap_disabled": env.get("HLE_DISABLE_ROUTER_AWARE_CHILD_WORKER_CAP"),
        "raw_content_persisted": False,
    }


def model_router_primary_key_present(env: dict[str, str]) -> bool:
    return any(str(env.get(name, "")).strip() for name in ("GPT5_API_KEY", "RUOLI_GPT_KEY", "OPENAI_API_KEY"))


def run_live_model_preflight(
    *,
    models: str,
    env: dict[str, str],
    timeout_sec: float = 60.0,
) -> dict[str, Any]:
    """Probe live model access before launching expensive shards."""
    model_names = [item.strip() for item in str(models or "").split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    if not model_router_primary_key_present(env):
        rows = [
            {
                "model": model,
                "ok": False,
                "error_type": "RuntimeError",
                "error_label": "missing GPT5_API_KEY, RUOLI_GPT_KEY, or OPENAI_API_KEY",
            }
            for model in model_names
        ]
        return {
            "preflight_kind": "hle_live_model_preflight",
            "passed": False,
            "models": model_names,
            "rows": rows,
            "raw_content_persisted": False,
        }

    probe_timeout = None if timeout_sec <= 0 else max(5.0, float(timeout_sec))
    per_attempt_timeout = None if timeout_sec <= 0 else max(1.0, min(float(timeout_sec), 30.0))
    for model in model_names:
        probe_env = env.copy()
        probe_env.setdefault("MODEL_ROUTER_SUBPROCESS_CALLS", "1")
        probe_env["MODEL_ROUTER_ATTEMPTS"] = "1"
        if per_attempt_timeout is None:
            probe_env["MODEL_ROUTER_TIMEOUT"] = "0"
            probe_env["MODEL_ROUTER_PER_ATTEMPT_TIMEOUT"] = "0"
        else:
            probe_env["MODEL_ROUTER_TIMEOUT"] = str(per_attempt_timeout)
            probe_env["MODEL_ROUTER_PER_ATTEMPT_TIMEOUT"] = str(per_attempt_timeout)
        script = (
            "import json, sys\n"
            "from assumption_os.hle_smoke_eval import _call_model\n"
            "cfg = json.loads(sys.stdin.read())\n"
            "text = _call_model(model=cfg['model'], prompt='Return exactly {\"answer\":\"A\"}.', "
            "timeout=cfg['timeout'], max_tokens=16)\n"
            "print('ok' if text.strip() else 'empty')\n"
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-c", script],
                input=json.dumps({"model": model, "timeout": per_attempt_timeout}),
                text=True,
                capture_output=True,
                cwd=str(Path.cwd()),
                env=probe_env,
                timeout=None if probe_timeout is None else probe_timeout + 5.0,
                check=False,
            )
        except Exception as exc:
            rows.append({
                "model": model,
                "ok": False,
                "error_type": type(exc).__name__,
                "error_label": _redact_model_preflight_error(str(exc), env),
            })
            continue
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        ok = completed.returncode == 0 and bool(stdout)
        rows.append({
            "model": model,
            "ok": ok,
            "returncode": int(completed.returncode),
            "stdout_hash": "" if not stdout else _stable_text_hash(stdout),
            "error_type": "" if ok else "RuntimeError",
            "error_label": "" if ok else _redact_model_preflight_error((stderr or stdout or "model_preflight_failed")[-240:], env),
        })
    return {
        "preflight_kind": "hle_live_model_preflight",
        "passed": all(row.get("ok") for row in rows) if rows else True,
        "models": model_names,
        "rows": rows,
        "raw_content_persisted": False,
    }


def _stable_text_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def _redact_model_preflight_error(text: str, env: dict[str, str]) -> str:
    redacted = str(text or "")
    for key_name in ("GPT5_API_KEY", "RUOLI_GPT_KEY", "OPENAI_API_KEY"):
        secret = env.get(key_name)
        if secret:
            redacted = redacted.replace(secret, "[redacted]")
    redacted = re.sub(r"Bearer\s+[A-Za-z0-9._:-]+", "Bearer [redacted]", redacted)
    redacted = re.sub(r"sk-[A-Za-z0-9._:-]+", "sk-[redacted]", redacted)
    return redacted[:240]


def _env_truthy(env: dict[str, str], name: str) -> bool:
    return str(env.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def apply_hle_offline_defaults(env: dict[str, str]) -> dict[str, str]:
    """Default HLE runs to local data and cache-only source retrieval when present."""
    if _env_truthy(env, "HLE_DISABLE_LOCAL_HLE_DEFAULTS"):
        return env

    dataset_path = Path(env.get("HLE_DATASET_LOCAL_PATH") or DEFAULT_HLE_DATASET_LOCAL_PATH)
    if not str(env.get("HLE_DATASET_LOCAL_PATH", "")).strip() and dataset_path.exists():
        env["HLE_DATASET_LOCAL_PATH"] = str(dataset_path)

    cache_path = Path(env.get("HLE_EVIDENCE_SOURCE_CACHE_DIR") or DEFAULT_HLE_EVIDENCE_SOURCE_CACHE_DIR)
    if not str(env.get("HLE_EVIDENCE_SOURCE_CACHE_DIR", "")).strip() and cache_path.exists():
        env["HLE_EVIDENCE_SOURCE_CACHE_DIR"] = str(cache_path)

    has_source_cache = bool(str(env.get("HLE_EVIDENCE_SOURCE_CACHE_DIR", "")).strip())
    explicit_source_policy = any(
        str(env.get(key, "")).strip()
        for key in (
            "HLE_EVIDENCE_SOURCE_CACHE_ONLY",
            "HLE_SOURCE_SEARCH_CACHE_ONLY",
            "HLE_DISABLE_LIVE_SOURCE_SEARCH",
        )
    )
    if has_source_cache and not explicit_source_policy and not _env_truthy(env, "HLE_ALLOW_LIVE_SOURCE_SEARCH"):
        env["HLE_EVIDENCE_SOURCE_CACHE_ONLY"] = "1"
    if (
        not str(env.get("HLE_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC", "")).strip()
        and not str(env.get("HLE_RECURSIVE_CHILD_BATCH_TOTAL_WAIT_SEC", "")).strip()
    ):
        env["HLE_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC"] = "180"
    return env


def apply_live_network_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Make live HLE runs network-stable unless the caller explicitly overrides.

    The live endpoint often fails fast with transient SSL EOF errors under
    bursty shard/child concurrency.  These defaults keep the paper-clean runner
    from turning endpoint noise into an apparent algorithm failure while still
    preserving caller control through the existing CLI flags.
    """
    if not bool(getattr(args, "execute_live", False)):
        return args
    if getattr(args, "model_router_attempts", None) is None:
        args.model_router_attempts = 8
    if getattr(args, "model_router_backoff_base_sec", None) is None:
        args.model_router_backoff_base_sec = 1.5
    if getattr(args, "model_router_global_concurrency", None) is None:
        workers = max(1, int(getattr(args, "parallel_workers", 1) or 1))
        args.model_router_global_concurrency = max(1, min(4, workers))
    if not getattr(args, "model_router_global_concurrency_dir", ""):
        eval_id = str(getattr(args, "eval_id", "hle_parallel_shard_eval")).replace(os.sep, "_")
        args.model_router_global_concurrency_dir = f"/tmp/assumption_agent_model_slots_{eval_id}"
    if getattr(args, "model_router_global_slot_ttl_sec", None) is None:
        args.model_router_global_slot_ttl_sec = 7200.0
    if getattr(args, "model_router_global_slot_wait_sec", None) is None:
        args.model_router_global_slot_wait_sec = 7200.0
    return args


def run_parallel_shards(
    *,
    root: Path,
    shard_states: list[ShardRunState],
    parallel_workers: int,
    heartbeat_path: Path,
    poll_interval_sec: float,
    heartbeat_interval_sec: float,
    soft_timeout_sec: float | None,
    terminate_grace_sec: float,
    kill_on_soft_timeout: bool,
    launch_stagger_sec: float,
    env: dict[str, str],
) -> list[ShardRunState]:
    if parallel_workers <= 0:
        raise ValueError("parallel_workers must be positive")
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    pending = [state for state in shard_states if state.status != "completed"]
    running: list[ShardRunState] = []
    completed: list[ShardRunState] = [state for state in shard_states if state.status == "completed"]
    last_heartbeat = 0.0
    while pending or running:
        now = time.monotonic()
        while pending and len(running) < parallel_workers:
            state = pending.pop(0)
            state.spec.out.parent.mkdir(parents=True, exist_ok=True)
            state.spec.md_out.parent.mkdir(parents=True, exist_ok=True)
            state.spec.log_out.parent.mkdir(parents=True, exist_ok=True)
            state._stdout_handle = state.spec.stdout_out.open("w", encoding="utf-8")
            try:
                state.process = subprocess.Popen(
                    state.command,
                    cwd=str(root),
                    env=env,
                    stdout=state._stdout_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                state.started_monotonic = time.monotonic()
                state.status = "running"
                running.append(state)
                if launch_stagger_sec > 0 and pending and len(running) < parallel_workers:
                    time.sleep(launch_stagger_sec)
            except Exception as exc:  # pragma: no cover - defensive subprocess path.
                state.error = f"{type(exc).__name__}: {exc}"
                state.status = "spawn_failed"
                state.finished_monotonic = time.monotonic()
                _close_stdout(state)
                completed.append(state)
        still_running: list[ShardRunState] = []
        for state in running:
            process = state.process
            if process is None:
                state.status = "spawn_failed"
                state.finished_monotonic = time.monotonic()
                completed.append(state)
                continue
            returncode = process.poll()
            elapsed = state.elapsed_sec(now)
            if returncode is None and soft_timeout_sec is not None and elapsed is not None:
                state.process_timeout_policy = "terminate_and_kill" if kill_on_soft_timeout else "watch_only"
                if elapsed > soft_timeout_sec:
                    state.soft_timeout_observed = True
                    if kill_on_soft_timeout:
                        if not state.soft_timeout_sent:
                            state.soft_timeout_sent = True
                            state.status = "soft_timed_out"
                            process.terminate()
                        elif elapsed > soft_timeout_sec + terminate_grace_sec and not state.hard_kill_sent:
                            state.hard_kill_sent = True
                            state.status = "hard_killed"
                            process.kill()
            returncode = process.poll()
            if returncode is None:
                still_running.append(state)
                continue
            state.returncode = int(returncode)
            state.finished_monotonic = time.monotonic()
            if state.status in {"soft_timed_out", "hard_killed"}:
                pass
            elif returncode == 0:
                state.status = "completed"
            else:
                state.status = "failed"
            _close_stdout(state)
            completed.append(state)
        running = still_running
        now = time.monotonic()
        if now - last_heartbeat >= heartbeat_interval_sec:
            heartbeat_path.write_text(
                json.dumps(build_heartbeat(shard_states), ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            last_heartbeat = now
        if pending or running:
            time.sleep(max(0.1, poll_interval_sec))
    heartbeat_path.write_text(
        json.dumps(build_heartbeat(shard_states), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return shard_states


def _close_stdout(state: ShardRunState) -> None:
    handle = state._stdout_handle
    if handle is not None:
        handle.flush()
        handle.close()
        state._stdout_handle = None


def build_heartbeat(states: list[ShardRunState]) -> dict[str, Any]:
    now = time.monotonic()
    wall_now = time.time()
    status_counts = Counter(state.status for state in states)
    shard_rows = []
    for state in states:
        latest_event = _read_latest_jsonl_event(state.spec.log_out)
        log_progress = _jsonl_progress_summary(state.spec.log_out, wall_now=wall_now)
        process_memory = _update_process_memory_snapshot(state)
        process_pid = state.process.pid if state.process is not None else None
        shard_rows.append(
            {
                "shard_index": state.spec.shard_index,
                "eval_id": state.spec.eval_id,
                "status": state.status,
                "returncode": state.returncode,
                "process_pid": process_pid,
                "process_memory": process_memory,
                "process_peak_rss_kb": state.peak_rss_kb,
                "process_peak_vms_kb": state.peak_vms_kb,
                "elapsed_sec": state.elapsed_sec(now),
                "sample_size": state.spec.sample_size,
                "seed_offset": state.spec.seed_offset,
                "out_exists": state.spec.out.exists(),
                "log_out_exists": state.spec.log_out.exists(),
                "soft_timeout_sent": state.soft_timeout_sent,
                "soft_timeout_observed": state.soft_timeout_observed,
                "hard_kill_sent": state.hard_kill_sent,
                "process_timeout_policy": state.process_timeout_policy,
                "jsonl_line_count": log_progress["line_count"],
                "jsonl_age_sec": log_progress["age_sec"],
                "latest_event": latest_event,
                "error": state.error,
            }
        )
    return {
        "heartbeat_kind": "hle_parallel_shard_runner",
        "status_counts": dict(sorted(status_counts.items())),
        "shards": shard_rows,
        "raw_content_persisted": False,
    }


def _update_process_memory_snapshot(state: ShardRunState) -> dict[str, Any]:
    pid = state.process.pid if state.process is not None else None
    snapshot = _process_memory_snapshot(pid)
    if snapshot:
        state.last_process_memory = snapshot
        rss_kb = snapshot.get("rss_kb")
        vms_kb = snapshot.get("vms_kb")
        if isinstance(rss_kb, int):
            state.peak_rss_kb = max(state.peak_rss_kb or 0, rss_kb)
        if isinstance(vms_kb, int):
            state.peak_vms_kb = max(state.peak_vms_kb or 0, vms_kb)
    return dict(state.last_process_memory)


def _process_memory_snapshot(pid: int | None) -> dict[str, Any]:
    if not pid:
        return {}
    status_path = Path("/proc") / str(pid) / "status"
    try:
        lines = status_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return {}
    fields: dict[str, Any] = {
        "pid": int(pid),
        "source": "proc_status",
    }
    status_keys = {
        "VmRSS": "rss_kb",
        "VmHWM": "hwm_kb",
        "VmSize": "vms_kb",
        "VmData": "data_kb",
        "VmSwap": "swap_kb",
        "Threads": "threads",
    }
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out_key = status_keys.get(key)
        if not out_key:
            continue
        match = re.search(r"\d+", value)
        if match:
            fields[out_key] = int(match.group(0))
    return fields


def write_preflight_heartbeat(
    path: Path,
    *,
    eval_id: str,
    phase: str,
    run_dir: Path,
    details: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "heartbeat_kind": "hle_parallel_shard_runner_preflight",
        "eval_id": eval_id,
        "phase": phase,
        "run_dir": str(run_dir),
        "timestamp_unix": round(time.time(), 3),
        "raw_content_persisted": False,
    }
    if details:
        payload["details"] = details
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _read_latest_jsonl_event(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    latest: dict[str, Any] | None = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                latest = {
                    "event": event.get("event"),
                    "model": event.get("model"),
                    "variant": event.get("variant"),
                    "problem_id_hash": event.get("problem_id_hash"),
                    "error_type": event.get("error_type"),
                    "stage": event.get("stage"),
                    "timestamp_utc": event.get("timestamp_utc"),
                }
    except OSError:
        return None
    return latest


def _jsonl_progress_summary(path: Path, *, wall_now: float) -> dict[str, Any]:
    if not path.exists():
        return {"line_count": 0, "age_sec": None}
    line_count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    line_count += 1
        age_sec = round(max(0.0, wall_now - path.stat().st_mtime), 4)
    except OSError:
        return {"line_count": line_count, "age_sec": None}
    return {"line_count": line_count, "age_sec": age_sec}


def load_shard_payloads(specs: list[ShardSpec]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.out.exists():
            continue
        try:
            payloads.append(json.loads(spec.out.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return payloads


def mark_reusable_completed_shards(states: list[ShardRunState]) -> dict[str, Any]:
    """Mark shards with an existing valid payload as completed.

    This lets expensive live HLE runs resume without rerunning already-scored
    hashes.  The payload is still loaded through ``load_shard_payloads`` during
    aggregation; this helper only prevents duplicate subprocess execution.
    """
    reused = 0
    missing_or_invalid = 0
    for state in states:
        payload = _load_existing_shard_payload(state.spec.out)
        if payload:
            state.status = "completed"
            state.returncode = 0
            state.reused_existing_payload = True
            reused += 1
        else:
            missing_or_invalid += 1
    return {
        "enabled": True,
        "reused_shard_count": reused,
        "pending_shard_count": missing_or_invalid,
        "raw_content_persisted": False,
    }


def _load_existing_shard_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if not payload.get("rows") and not payload.get("run_rows"):
        return None
    if payload.get("metrics", {}).get("raw_content_persisted") is not False:
        return None
    return payload


def aggregate_parallel_payload(
    *,
    eval_id: str,
    specs: list[ShardSpec],
    states: list[ShardRunState],
    shard_payloads: list[dict[str, Any]],
    execute_live: bool,
    models: str,
    variants: str,
    total_sample_size: int,
    shard_size: int,
    parallel_workers: int,
    soft_timeout_sec: float | None,
    kill_on_soft_timeout: bool = False,
    shard_sample_dedupe: dict[str, Any] | None = None,
    reuse_completed_shards: dict[str, Any] | None = None,
    launch_stagger_sec: float = 0.0,
    diagnostic_log_out: Path | None = None,
    model_router_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_rows = _merged_run_rows(shard_payloads)
    metrics = _parallel_metrics(run_rows=run_rows, shard_payloads=shard_payloads)
    error_stratification = build_error_stratification(
        rows=run_rows,
        specs=specs,
        states=states,
    )
    pollution_audit = build_pollution_audit(
        rows=run_rows,
        shard_payloads=shard_payloads,
        metrics=metrics,
        error_stratification=error_stratification,
        execute_live=execute_live,
    )
    model_budget_fairness_audit = build_model_budget_fairness_audit(rows=run_rows)
    failure_diagnostics = build_failure_diagnostics(rows=run_rows)
    fair_baseline_gate = _agent_meets_best_control_gate(metrics)
    operator_activation = metrics.get("operator_activation_summary", {})
    operator_application = metrics.get("operator_application_summary", {})
    gates = {
        "all_shards_finished_without_process_failure": all(
            state.status == "completed" for state in states
        ),
        "all_available_payloads_preserve_raw_content": all(
            (payload.get("metrics") or {}).get("raw_content_persisted") is False
            for payload in shard_payloads
        ),
        "sample_rows_loaded": metrics["sample_count"] >= min(total_sample_size, 1),
        "requested_sample_rows_loaded": metrics["sample_count"] >= total_sample_size,
        "live_rows_resolved_if_requested": (
            not execute_live
            or metrics["resolved_live_model_calls"] == metrics["planned_live_model_calls"]
        ),
        "agent_not_below_best_same_model_control": fair_baseline_gate["passed"],
        "assumption_operator_activated_if_selected": bool(operator_activation.get("passed", True)),
        "operator_application_fidelity_if_verified": (
            int(operator_application.get("verifier_activated_count") or 0) == 0
            or bool(operator_application.get("passed", True))
        ),
    }
    paper_clean_gates = dict(gates)
    paper_clean_gates["zero_top_level_live_errors"] = error_stratification["top_level_error_count"] == 0
    paper_clean_gates["zero_process_timeouts"] = error_stratification["process_timeout_count"] == 0
    paper_clean_gates["no_duplicate_sample_problems"] = metrics["duplicate_sample_problem_count"] == 0
    paper_clean_gates.update(model_budget_fairness_audit["gates"])
    pollution_gates = pollution_audit["gates"]
    return {
        "eval_id": eval_id,
        "eval_kind": "hle_parallel_shard_runner",
        "dataset": DATASET_NAME,
        "official_sources": HLE_OFFICIAL_SOURCES,
        "performance_validation": True,
        "validation_scope": (
            "Runs HLE smoke-eval shards through a bounded parallel subprocess runner. "
            "The artifact stores only hashes, counts, process states, and error types."
        ),
        "sampling": {
            "requested_total_sample_size": total_sample_size,
            "shard_size": shard_size,
            "planned_shard_count": len(specs),
            "parallel_workers": parallel_workers,
            "models": [item.strip() for item in models.split(",") if item.strip()],
            "variants": [item.strip() for item in variants.split(",") if item.strip()],
            "shard_sample_dedupe": shard_sample_dedupe or {"enabled": False},
        },
        "runtime_policy": {
            "execute_live": execute_live,
            "soft_timeout_sec": soft_timeout_sec,
            "process_timeout_policy": "terminate_and_kill" if kill_on_soft_timeout else "watch_only",
            "kill_on_soft_timeout": kill_on_soft_timeout,
            "launch_stagger_sec": launch_stagger_sec,
            "reuse_completed_shards": reuse_completed_shards or {"enabled": False},
            "model_router": model_router_policy or {},
            "raw_content_persisted": False,
        },
        "diagnostic_log_out": str(diagnostic_log_out) if diagnostic_log_out else None,
        "logging_policy": {
            "event_stream": "jsonl",
            "raw_content_persisted": False,
            "prediction_text_persisted": False,
            "gold_answer_persisted": False,
            "event_granularity": "runner lifecycle, shard states, payload load counts, aggregate gates",
        },
        "shards": [_shard_summary(state) for state in states],
        "loaded_shard_payload_count": len(shard_payloads),
        "metrics": metrics,
        "error_stratification": error_stratification,
        "pollution_audit": pollution_audit,
        "model_budget_fairness_audit": model_budget_fairness_audit,
        "fair_baseline_gate": fair_baseline_gate,
        "failure_diagnostics": failure_diagnostics,
        "pass": all(gates.values()),
        "paper_clean_pass": all(paper_clean_gates.values()),
        "pollution_pass": all(pollution_gates.values()),
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "paper_clean_failed_gates": [name for name, passed in paper_clean_gates.items() if not passed],
        "pollution_failed_gates": [name for name, passed in pollution_gates.items() if not passed],
        "raw_content_persisted": False,
    }


def _shard_summary(state: ShardRunState) -> dict[str, Any]:
    return {
        "shard_index": state.spec.shard_index,
        "eval_id": state.spec.eval_id,
        "status": state.status,
        "returncode": state.returncode,
        "process_peak_rss_kb": state.peak_rss_kb,
        "process_peak_vms_kb": state.peak_vms_kb,
        "last_process_memory": dict(state.last_process_memory),
        "elapsed_sec": state.elapsed_sec(),
        "sample_size": state.spec.sample_size,
        "seed_offset": state.spec.seed_offset,
        "out": str(state.spec.out),
        "log_out": str(state.spec.log_out),
        "stdout_out": str(state.spec.stdout_out),
        "soft_timeout_sent": state.soft_timeout_sent,
        "soft_timeout_observed": state.soft_timeout_observed,
        "hard_kill_sent": state.hard_kill_sent,
        "reused_existing_payload": state.reused_existing_payload,
        "process_timeout_policy": state.process_timeout_policy,
        "error": state.error,
    }


def _merged_run_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        rows.extend(payload.get("rows", []) or payload.get("run_rows", []) or [])
    return rows


def _parallel_metrics(*, run_rows: list[dict[str, Any]], shard_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        by_key[f"{row.get('model')}::{row.get('variant')}"].append(row)
    sample_count = sum(int((payload.get("metrics") or {}).get("sample_count") or 0) for payload in shard_payloads)
    planned = sum(
        int((payload.get("metrics") or {}).get("planned_live_model_calls") or 0)
        for payload in shard_payloads
    )
    live_executed = sum(
        int((payload.get("metrics") or {}).get("live_model_calls_executed") or 0)
        for payload in shard_payloads
    )
    underlying = sum(
        int((payload.get("metrics") or {}).get("underlying_model_calls_executed") or 0)
        for payload in shard_payloads
    )
    resolved = sum(
        int((payload.get("metrics") or {}).get("resolved_live_model_calls") or 0)
        for payload in shard_payloads
    )
    sample_problem_hashes = _merged_sample_problem_hashes(shard_payloads)
    row_problem_hashes = [str(row.get("problem_id_hash")) for row in run_rows if row.get("problem_id_hash")]
    distinct_sample_problem_hashes = set(sample_problem_hashes)
    distinct_row_problem_hashes = set(row_problem_hashes)
    return {
        "sample_count": sample_count,
        "distinct_sample_problem_count": len(distinct_sample_problem_hashes),
        "duplicate_sample_problem_count": max(0, len(sample_problem_hashes) - len(distinct_sample_problem_hashes)),
        "distinct_scored_problem_count": len(distinct_row_problem_hashes),
        "duplicate_scored_problem_count": max(0, len(row_problem_hashes) - len(distinct_row_problem_hashes)),
        "planned_live_model_calls": planned,
        "live_model_calls_executed": live_executed,
        "underlying_model_calls_executed": underlying,
        "resolved_live_model_calls": resolved,
        "scored_row_count": len(run_rows),
        "overall_accuracy": _accuracy(run_rows),
        "by_model_variant": {key: _aggregate_rows(rows) for key, rows in sorted(by_key.items())},
        "control_comparison": _control_comparison(run_rows),
        "module_activation_summary": _module_activation_summary(run_rows),
        "expected_but_missing_modules": _expected_but_missing_modules(run_rows),
        "component_efficacy_summary": _component_efficacy_summary(run_rows),
        "operator_activation_summary": _operator_activation_summary(run_rows),
        "operator_application_summary": _operator_application_summary(run_rows),
        "route_credit_table": _route_credit_table(run_rows),
        "clean_shared_subset": _clean_shared_subset(run_rows),
        "raw_content_persisted": False,
    }


def _merged_sample_problem_hashes(payloads: list[dict[str, Any]]) -> list[str]:
    hashes: list[str] = []
    for payload in payloads:
        sampling = payload.get("sampling") or {}
        for value in sampling.get("sample_problem_hashes", []) or []:
            hashes.append(str(value))
    return hashes


def _accuracy(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return round(sum(1 for row in rows if row.get("correct")) / len(rows), 4)


def _clean_shared_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_model[str(row.get("model"))][str(row.get("variant"))][str(row.get("problem_id_hash"))] = row
    out: dict[str, Any] = {}
    for model, by_variant in sorted(by_model.items()):
        if not by_variant:
            continue
        shared_ids: set[str] | None = None
        for variant_rows in by_variant.values():
            clean_ids = {pid for pid, row in variant_rows.items() if not row.get("error")}
            shared_ids = clean_ids if shared_ids is None else shared_ids & clean_ids
        shared_ids = shared_ids or set()
        variant_metrics = {
            variant: {
                "n": len(shared_ids),
                "accuracy": _accuracy([variant_rows[pid] for pid in sorted(shared_ids)]),
            }
            for variant, variant_rows in sorted(by_variant.items())
        }
        out[model] = {
            "shared_clean_problem_count": len(shared_ids),
            "by_variant": variant_metrics,
        }
    return out


def build_model_budget_fairness_audit(*, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit whether Agent rows have fair model and call-budget controls.

    This is a paper-clean gate, not a runner-health gate.  The runner can still
    aggregate unfair experiments, but paper-facing claims must disclose and
    satisfy same-model, strong-baseline, and budget-matched controls whenever
    the Agent uses a stronger/different model or more calls.
    """
    present_by_model: dict[str, set[str]] = defaultdict(set)
    row_count_by_model_variant: Counter[str] = Counter()
    for row in rows:
        model = _clean_model_name(row.get("model"))
        variant = str(row.get("variant") or "")
        if not model or not variant:
            continue
        present_by_model[model].add(variant)
        row_count_by_model_variant[f"{model}::{variant}"] += 1

    agent_rows = [row for row in rows if str(row.get("variant") or "").startswith("assumption_agent")]
    agent_top_models = sorted({
        model for model in (_clean_model_name(row.get("model")) for row in agent_rows) if model
    })
    metadata_complete = all(isinstance(row.get("component_efficacy"), dict) and bool(row.get("component_efficacy")) for row in agent_rows)

    effective_records: list[dict[str, Any]] = []
    stronger_or_different_models: set[str] = set()
    budget_target_models: set[str] = set()
    multi_call_agent_row_count = 0
    for row in agent_rows:
        top_model = _clean_model_name(row.get("model"))
        entries = _agent_effective_model_entries(row)
        different_models = sorted({
            entry["model"]
            for entry in entries
            if entry.get("model") and top_model and entry["model"] != top_model
        })
        stronger_or_different_models.update(different_models)
        multi_call = _agent_row_uses_more_than_single_call(row)
        if multi_call:
            multi_call_agent_row_count += 1
            budget_target_models.update(different_models or ([top_model] if top_model else []))
        effective_records.append({
            "problem_id_hash": row.get("problem_id_hash"),
            "variant": row.get("variant"),
            "top_model": top_model,
            "effective_models": entries,
            "different_effective_models": different_models,
            "multi_call_detected": multi_call,
        })

    missing_same_model_controls = _missing_controls(
        present_by_model=present_by_model,
        models=agent_top_models,
        required_variants=PAPER_CLEAN_STANDARD_CONTROL_VARIANTS,
    )
    missing_strong_baseline_controls = _missing_controls(
        present_by_model=present_by_model,
        models=sorted(stronger_or_different_models),
        required_variants=PAPER_CLEAN_STANDARD_CONTROL_VARIANTS,
    )
    missing_budget_matched_controls = _missing_controls(
        present_by_model=present_by_model,
        models=sorted(budget_target_models),
        required_variants=PAPER_CLEAN_BUDGET_MATCHED_CONTROL_VARIANTS,
    )

    gates = {
        "model_budget_metadata_complete": not agent_rows or metadata_complete,
        "same_model_controls_present": not agent_rows or not missing_same_model_controls,
        "strong_baseline_controls_present_if_needed": (
            not stronger_or_different_models or not missing_strong_baseline_controls
        ),
        "budget_matched_controls_present_if_needed": (
            multi_call_agent_row_count == 0 or not missing_budget_matched_controls
        ),
    }
    gates["model_budget_fairness_accounted"] = all(gates.values())
    return {
        "audit_kind": "hle_model_budget_fairness_audit",
        "agent_row_count": len(agent_rows),
        "agent_top_models": agent_top_models,
        "stronger_or_different_effective_models": sorted(stronger_or_different_models),
        "multi_call_agent_row_count": multi_call_agent_row_count,
        "budget_target_models": sorted(budget_target_models),
        "required_standard_control_variants": list(PAPER_CLEAN_STANDARD_CONTROL_VARIANTS),
        "required_budget_matched_control_variants": list(PAPER_CLEAN_BUDGET_MATCHED_CONTROL_VARIANTS),
        "present_variants_by_model": {
            model: sorted(variants)
            for model, variants in sorted(present_by_model.items())
        },
        "row_count_by_model_variant": dict(sorted(row_count_by_model_variant.items())),
        "missing_same_model_controls": missing_same_model_controls,
        "missing_strong_baseline_controls": missing_strong_baseline_controls,
        "missing_budget_matched_controls": missing_budget_matched_controls,
        "agent_effective_model_records": effective_records[:20],
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "raw_content_persisted": False,
    }


def _missing_controls(
    *,
    present_by_model: dict[str, set[str]],
    models: list[str],
    required_variants: tuple[str, ...],
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for model in models:
        present = present_by_model.get(model, set())
        absent = [variant for variant in required_variants if variant not in present]
        if absent:
            missing.append({
                "model": model,
                "missing_variants": absent,
                "present_variants": sorted(present),
            })
    return missing


def _clean_model_name(value: Any) -> str:
    return str(value or "").strip()


def _agent_effective_model_entries(row: dict[str, Any]) -> list[dict[str, str]]:
    model = _clean_model_name(row.get("model"))
    entries: list[dict[str, str]] = []
    if model:
        entries.append({"source": "top_level", "model": model})
    ce = row.get("component_efficacy") if isinstance(row.get("component_efficacy"), dict) else {}
    for section_name, model_key in (
        ("recursive", "child_model"),
        ("child_model", "child_model"),
        ("critic_model", "critic_model"),
        ("recursive_timeout_recovery", "recovery_model"),
        ("child_model_failover", "failed_child_model"),
        ("critic_synthesis_child", "critic_model"),
    ):
        section = ce.get(section_name) if isinstance(ce.get(section_name), dict) else {}
        candidate = _clean_model_name(section.get(model_key))
        if candidate:
            entries.append({"source": section_name, "model": candidate})
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = (entry["source"], entry["model"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _agent_row_uses_more_than_single_call(row: dict[str, Any]) -> bool:
    ce = row.get("component_efficacy") if isinstance(row.get("component_efficacy"), dict) else {}
    recursive = ce.get("recursive") if isinstance(ce.get("recursive"), dict) else {}
    selection = ce.get("selection") if isinstance(ce.get("selection"), dict) else {}
    if int(recursive.get("planned_child_count") or 0) > 1:
        return True
    if int(recursive.get("child_count") or 0) > 1:
        return True
    if int(recursive.get("answered_child_count") or 0) > 1:
        return True
    if bool(selection.get("verifier_model_call")):
        return True
    for section_name in (
        "claim_verifier",
        "domain_rule_mc_verifier",
        "mc_option_evidence_scorer",
        "critic_synthesis_child",
        "mc_option_sweep_candidates",
        "counter_assumption_challenge",
        "recursive_timeout_recovery",
        "child_model_failover",
    ):
        section = ce.get(section_name) if isinstance(ce.get(section_name), dict) else {}
        if section.get("status") == "activated":
            return True
    flags = ce.get("flags") if isinstance(ce.get("flags"), dict) else {}
    return any(
        bool(flags.get(key))
        for key in (
            "recursive_child_validation_activated",
            "claim_verifier_activated",
            "domain_rule_mc_verifier_activated",
            "mc_option_evidence_scorer_activated",
            "critic_synthesis_activated",
            "mc_option_sweep_activated",
            "counter_assumption_challenge_activated",
        )
    )


def build_error_stratification(
    *,
    rows: list[dict[str, Any]],
    specs: list[ShardSpec],
    states: list[ShardRunState],
) -> dict[str, Any]:
    top_level_by_variant: Counter[str] = Counter()
    top_level_by_type: Counter[str] = Counter()
    top_level_by_variant_type: Counter[str] = Counter()
    top_level_by_label: Counter[str] = Counter()
    top_level_by_variant_label: Counter[str] = Counter()
    for row in rows:
        error = row.get("error") or {}
        if not error:
            continue
        variant = str(row.get("variant"))
        error_type = str(error.get("type") or "unknown")
        error_label = _sanitize_error_label(error.get("message") or error_type)
        top_level_by_variant[variant] += 1
        top_level_by_type[error_type] += 1
        top_level_by_variant_type[f"{variant}::{error_type}"] += 1
        top_level_by_label[error_label] += 1
        top_level_by_variant_label[f"{variant}::{error_label}"] += 1

    jsonl_events = _jsonl_error_events(specs)
    process_status_counts = Counter(state.status for state in states)
    process_timeout_count = sum(1 for state in states if state.soft_timeout_sent or state.hard_kill_sent)
    return {
        "top_level_error_count": sum(top_level_by_variant.values()),
        "top_level_errors_by_variant": dict(sorted(top_level_by_variant.items())),
        "top_level_errors_by_type": dict(sorted(top_level_by_type.items())),
        "top_level_errors_by_variant_type": dict(sorted(top_level_by_variant_type.items())),
        "top_level_errors_by_label": dict(sorted(top_level_by_label.items())),
        "top_level_errors_by_variant_label": dict(sorted(top_level_by_variant_label.items())),
        "jsonl_error_event_count": sum(jsonl_events["by_event"].values()),
        "jsonl_error_events_by_event": dict(sorted(jsonl_events["by_event"].items())),
        "jsonl_error_events_by_variant": dict(sorted(jsonl_events["by_variant"].items())),
        "jsonl_error_events_by_type": dict(sorted(jsonl_events["by_error_type"].items())),
        "jsonl_error_events_by_label": dict(sorted(jsonl_events["by_error_label"].items())),
        "process_status_counts": dict(sorted(process_status_counts.items())),
        "process_timeout_count": process_timeout_count,
        "raw_content_persisted": False,
    }


def build_pollution_audit(
    *,
    rows: list[dict[str, Any]],
    shard_payloads: list[dict[str, Any]],
    metrics: dict[str, Any],
    error_stratification: dict[str, Any],
    execute_live: bool,
) -> dict[str, Any]:
    sample_hashes = _merged_sample_problem_hashes(shard_payloads)
    api_summaries = [payload.get("api_summary") or {} for payload in shard_payloads]
    excluded_existing_problem_count = sum(
        int(summary.get("excluded_existing_problem_count") or 0)
        for summary in api_summaries
    )
    exclude_existing_enabled_count = sum(
        1 for summary in api_summaries if bool(summary.get("exclude_existing_hle_artifacts"))
    )
    context_by_variant = _context_pollution_by_variant(rows)
    selection_credit = _selection_credit(rows)
    clean_shared = metrics.get("clean_shared_subset") or {}
    agent_advantage = _clean_shared_agent_advantage(clean_shared)
    top_level_errors = int(error_stratification.get("top_level_error_count") or 0)
    process_timeouts = int(error_stratification.get("process_timeout_count") or 0)
    clean_shared_problem_count = max(
        [int(row.get("shared_clean_problem_count") or 0) for row in clean_shared.values()] or [0]
    )
    claim_scope = {
        "paper_clean_claim_allowed": top_level_errors == 0 and process_timeouts == 0,
        "selective_agent_advantage_claim_allowed": bool(agent_advantage.get("agent_beats_all_controls")),
        "recommended_hle_claim_scope": (
            "full_resolved_rows"
            if top_level_errors == 0 and process_timeouts == 0
            else "clean_shared_subset_due_to_endpoint_noise"
        ),
        "agent_advantage": agent_advantage,
    }
    gates = {
        "raw_content_not_persisted": metrics.get("raw_content_persisted") is False,
        "fresh_problem_hashes_accounted": bool(sample_hashes) or not execute_live,
        "no_duplicate_problem_hashes": int(metrics.get("duplicate_sample_problem_count") or 0) == 0,
        "cache_live_separation_accounted": True,
        "endpoint_errors_separated": "top_level_errors_by_variant" in error_stratification,
        "clean_shared_subset_available_if_endpoint_errors": top_level_errors == 0 or clean_shared_problem_count > 0,
        "context_pollution_accounted": isinstance(context_by_variant, dict),
        "selection_credit_accounted": bool(selection_credit.get("by_selection_method")) or not rows,
        "claim_scope_downgraded_when_endpoint_errors": (
            top_level_errors == 0
            or claim_scope["recommended_hle_claim_scope"] == "clean_shared_subset_due_to_endpoint_noise"
        ),
    }
    return {
        "audit_kind": "hle_anti_pollution_audit",
        "fresh_problem_hash_exclusion": {
            "sample_problem_hash_count": len(sample_hashes),
            "distinct_sample_problem_hash_count": len(set(sample_hashes)),
            "duplicate_sample_problem_hash_count": max(0, len(sample_hashes) - len(set(sample_hashes))),
            "exclude_existing_enabled_shard_count": exclude_existing_enabled_count,
            "excluded_existing_problem_count": excluded_existing_problem_count,
        },
        "cache_live_separation": {
            "execute_live": execute_live,
            "planned_live_model_calls": metrics.get("planned_live_model_calls"),
            "resolved_live_model_calls": metrics.get("resolved_live_model_calls"),
            "live_model_calls_executed": metrics.get("live_model_calls_executed"),
            "underlying_model_calls_executed": metrics.get("underlying_model_calls_executed"),
            "top_level_error_count": top_level_errors,
            "process_timeout_count": process_timeouts,
            "top_level_errors_by_variant": error_stratification.get("top_level_errors_by_variant") or {},
        },
        "context_pollution": {
            "by_variant": context_by_variant,
            "summary": _context_pollution_summary(context_by_variant),
        },
        "module_credit_assignment": selection_credit,
        "claim_guard": claim_scope,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "raw_content_persisted": False,
    }


def _context_pollution_by_variant(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_variant: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        variant = str(row.get("variant") or "unknown")
        ce = row.get("component_efficacy") if isinstance(row.get("component_efficacy"), dict) else {}
        flags = ce.get("flags") if isinstance(ce.get("flags"), dict) else {}
        correct = bool(row.get("correct"))
        has_error = bool(row.get("error"))
        outcome_bucket = "error" if has_error else "correct" if correct else "wrong"
        graph = ce.get("graph") if isinstance(ce.get("graph"), dict) else {}
        evidence = ce.get("evidence") if isinstance(ce.get("evidence"), dict) else {}
        hipporag = ce.get("agent_hipporag") if isinstance(ce.get("agent_hipporag"), dict) else {}
        morphism = ce.get("morphism") if isinstance(ce.get("morphism"), dict) else {}
        if _flag_true(flags, "graph_context_discarded"):
            by_variant[variant]["graph_context_discarded"] += 1
        if _flag_true(flags, "generic_graph_context_only"):
            by_variant[variant]["generic_graph_context_only"] += 1
        if graph.get("status") in {"activated", "used"}:
            by_variant[variant]["graph_retrieval_activated"] += 1
            if _is_generic_harness_graph_context(graph):
                by_variant[variant]["graph_generic_harness_retrieved"] += 1
        if _flag_true(flags, "graph_context_injected"):
            by_variant[variant]["graph_context_used"] += 1
            by_variant[variant][f"graph_context_{outcome_bucket}"] += 1
            if _is_generic_harness_graph_context(graph):
                by_variant[variant]["graph_generic_harness_context"] += 1
        if _flag_true(flags, "evidence_bridge_activated") or evidence.get("status") in {"activated", "used"}:
            by_variant[variant]["evidence_context_used"] += 1
            by_variant[variant][f"evidence_context_{outcome_bucket}"] += 1
        if evidence.get("status") in {"no_results", "empty"}:
            by_variant[variant]["evidence_no_results"] += 1
        if _flag_true(flags, "agent_hipporag_context_activated") or hipporag.get("status") in {"activated", "used"}:
            by_variant[variant]["hipporag_context_used"] += 1
            by_variant[variant][f"hipporag_context_{outcome_bucket}"] += 1
        if hipporag.get("status") in {"no_results", "empty"}:
            by_variant[variant]["hipporag_no_results"] += 1
        if _flag_true(flags, "morphism_hit") or int(morphism.get("formal_hit_count") or 0) > 0 or int(morphism.get("structural_hit_count") or 0) > 0:
            by_variant[variant]["morphism_hit"] += 1
            by_variant[variant][f"morphism_{outcome_bucket}"] += 1
        if _flag_true(flags, "strong_morphism_hit") or int(morphism.get("strong_hit_count") or 0) > 0:
            by_variant[variant]["strong_morphism_hit"] += 1
            by_variant[variant][f"strong_morphism_{outcome_bucket}"] += 1
    return {variant: dict(sorted(counter.items())) for variant, counter in sorted(by_variant.items())}


def _context_pollution_summary(context_by_variant: dict[str, dict[str, int]]) -> dict[str, int]:
    summary: Counter[str] = Counter()
    for counts in context_by_variant.values():
        for key, value in counts.items():
            summary[key] += int(value)
    return dict(sorted(summary.items()))


def _flag_true(flags: dict[str, Any], key: str) -> bool:
    return bool(flags.get(key))


def _is_generic_harness_graph_context(graph: dict[str, Any]) -> bool:
    counts = graph.get("top_node_type_counts")
    if not isinstance(counts, dict) or not counts:
        return False
    total = sum(int(value or 0) for value in counts.values())
    harness = int(counts.get("harness") or counts.get("generic_harness") or 0)
    return total > 0 and harness >= total


def _selection_credit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, Counter[str]] = defaultdict(Counter)
    by_variant_method: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        method = _selection_method(row)
        variant = str(row.get("variant") or "unknown")
        correct = bool(row.get("correct"))
        has_error = bool(row.get("error"))
        by_method[method]["n"] += 1
        by_variant_method[f"{variant}::{method}"]["n"] += 1
        if correct:
            by_method[method]["correct"] += 1
            by_variant_method[f"{variant}::{method}"]["correct"] += 1
        if has_error:
            by_method[method]["error"] += 1
            by_variant_method[f"{variant}::{method}"]["error"] += 1
    return {
        "by_selection_method": _credit_counter_rows(by_method),
        "by_variant_selection_method": _credit_counter_rows(by_variant_method),
    }


def _credit_counter_rows(counters: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, counter in sorted(counters.items()):
        n = int(counter.get("n") or 0)
        out[key] = {
            "n": n,
            "correct": int(counter.get("correct") or 0),
            "error": int(counter.get("error") or 0),
            "accuracy": round(int(counter.get("correct") or 0) / n, 4) if n else None,
        }
    return out


def _selection_method(row: dict[str, Any]) -> str:
    ce = row.get("component_efficacy") if isinstance(row.get("component_efficacy"), dict) else {}
    selection = ce.get("selection") if isinstance(ce.get("selection"), dict) else {}
    stages = ce.get("stages") if isinstance(ce.get("stages"), dict) else {}
    multi = stages.get("multi_candidate_self_verifier") if isinstance(stages.get("multi_candidate_self_verifier"), dict) else {}
    metadata = row.get("call_metadata") if isinstance(row.get("call_metadata"), dict) else {}
    return str(
        selection.get("method")
        or selection.get("selection_method")
        or ce.get("selection_method")
        or multi.get("selection_method")
        or metadata.get("selection_method")
        or "unknown"
    )


def _clean_shared_agent_advantage(clean_shared: dict[str, Any]) -> dict[str, Any]:
    best_payload: dict[str, Any] = {
        "agent_beats_all_controls": False,
        "model": None,
        "agent_variant": None,
        "agent_accuracy": None,
        "best_control_accuracy": None,
        "margin": None,
    }
    for model, row in sorted(clean_shared.items()):
        by_variant = row.get("by_variant") or {}
        agent_items = [
            (variant, variant_row)
            for variant, variant_row in by_variant.items()
            if str(variant).startswith("assumption_agent")
        ]
        control_items = [
            (variant, variant_row)
            for variant, variant_row in by_variant.items()
            if not str(variant).startswith("assumption_agent")
        ]
        for agent_variant, agent_row in agent_items:
            agent_acc = agent_row.get("accuracy")
            control_accs = [
                control_row.get("accuracy")
                for _, control_row in control_items
                if control_row.get("accuracy") is not None
            ]
            if agent_acc is None or not control_accs:
                continue
            best_control = max(float(value) for value in control_accs)
            margin = round(float(agent_acc) - best_control, 4)
            if best_payload["margin"] is None or margin > float(best_payload["margin"]):
                best_payload = {
                    "agent_beats_all_controls": margin > 0,
                    "model": model,
                    "agent_variant": agent_variant,
                    "agent_accuracy": agent_acc,
                    "best_control_accuracy": best_control,
                    "margin": margin,
                }
    return best_payload


def build_failure_diagnostics(*, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model_problem: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_model_problem[str(row.get("model"))][str(row.get("problem_id_hash"))][str(row.get("variant"))] = row

    by_variant_answer_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_variant_domain: dict[str, Counter[str]] = defaultdict(Counter)
    agent_failure_buckets: Counter[str] = Counter()
    agent_gain_loss: Counter[str] = Counter()
    agent_selection_buckets: Counter[str] = Counter()
    verified_gate_buckets: Counter[str] = Counter()
    agent_problem_count = 0

    for row in rows:
        variant = str(row.get("variant") or "unknown")
        answer_type = str(row.get("answer_type") or "unknown")
        domain = _diagnostic_domain(row)
        outcome = "correct" if row.get("correct") else "wrong_or_error"
        by_variant_answer_type[variant][f"{answer_type}::{outcome}"] += 1
        by_variant_domain[variant][f"{domain}::{outcome}"] += 1

    for model, by_problem in sorted(by_model_problem.items()):
        del model
        for _, by_variant in sorted(by_problem.items()):
            agent_row = _first_agent_row(by_variant)
            if not agent_row:
                continue
            agent_problem_count += 1
            raw_row = by_variant.get("raw")
            hippo_row = by_variant.get("hipporag_baseline") or by_variant.get("hipporag")
            method = _selection_method(agent_row)
            agent_selection_buckets[method] += 1
            gate_status = _verified_gate_status(agent_row)
            verified_gate_buckets[gate_status] += 1
            agent_correct = bool(agent_row.get("correct"))
            raw_correct = bool(raw_row and raw_row.get("correct"))
            hippo_correct = bool(hippo_row and hippo_row.get("correct"))
            if agent_correct and not raw_correct:
                agent_gain_loss["agent_correct_raw_wrong"] += 1
            if raw_correct and not agent_correct:
                agent_gain_loss["raw_correct_agent_wrong_regression"] += 1
            if not agent_correct and raw_row and not raw_correct:
                agent_gain_loss["raw_also_wrong_agent_no_gain"] += 1
            if not agent_correct and hippo_row and not hippo_correct:
                agent_gain_loss["hipporag_also_wrong_agent_no_gain"] += 1
            if not agent_correct and raw_row and hippo_row and not raw_correct and not hippo_correct:
                agent_gain_loss["all_three_wrong"] += 1
            if agent_correct and raw_row and hippo_row and not raw_correct and not hippo_correct:
                agent_gain_loss["agent_only_correct"] += 1
            if agent_correct:
                continue
            agent_failure_buckets["agent_wrong_or_error"] += 1
            if agent_row.get("error"):
                agent_failure_buckets["agent_endpoint_error"] += 1
            if str(agent_row.get("answer_type")) == "exactMatch" and _diagnostic_domain(agent_row) == "math":
                agent_failure_buckets["math_exact_failed"] += 1
            if str(agent_row.get("answer_type")) == "multipleChoice":
                agent_failure_buckets["multiple_choice_selection_failed"] += 1
            flags = _row_flags(agent_row)
            if flags.get("candidate_generation_missed_gold"):
                agent_failure_buckets["candidate_generation_missed_gold"] += 1
            if flags.get("candidate_generation_missed_gold_with_sweep_coverage"):
                agent_failure_buckets["candidate_generation_missed_gold_with_sweep_coverage"] += 1
            if flags.get("missing_model_option_source_retry_scheduled"):
                agent_failure_buckets["missing_model_option_source_retry_unhelpful"] += 1
            if flags.get("mc_option_claim_source_verifier_cross_selection_blocked"):
                agent_failure_buckets["source_verifier_cross_selection_blocked"] += 1
            if (
                flags.get("gold_option_source_verifier_attempted")
                and not flags.get("gold_option_source_verifier_accepted")
            ):
                agent_failure_buckets["gold_option_source_verifier_unaccepted"] += 1
            if flags.get("gold_option_source_verifier_direct_source_insufficient"):
                agent_failure_buckets["gold_option_direct_source_insufficient"] += 1
            if flags.get("gold_option_source_verifier_indirect_or_generic"):
                agent_failure_buckets["gold_option_source_indirect_or_generic"] += 1
            if flags.get("evidence_bridge_activated") or flags.get("evidence_child_executed"):
                agent_failure_buckets["evidence_invalid_or_unhelpful"] += 1
            if flags.get("agent_hipporag_context_activated") or flags.get("agent_hipporag_child_executed"):
                agent_failure_buckets["hipporag_context_invalid_or_unhelpful"] += 1
            if (
                flags.get("morphism_hit")
                and not flags.get("strong_morphism_hit")
                and flags.get("morphism_context_injected")
            ):
                agent_failure_buckets["weak_morphism_unhelpful"] += 1
            elif flags.get("morphism_hit") and not flags.get("strong_morphism_hit"):
                agent_failure_buckets["weak_morphism_routing_only_not_credited"] += 1
            if _selection_is_verifier_like(method):
                agent_failure_buckets["verifier_or_arbitrator_wrong"] += 1
            if flags.get("majority_only_selection"):
                agent_failure_buckets["unverified_majority_wrong"] += 1
            if flags.get("verified_or_abstain_abstained"):
                agent_failure_buckets["verified_or_abstain_fallback_wrong"] += 1

    return {
        "diagnostic_kind": "hle_failure_diagnostics",
        "agent_problem_count": agent_problem_count,
        "by_variant_answer_type": {
            variant: dict(sorted(counter.items()))
            for variant, counter in sorted(by_variant_answer_type.items())
        },
        "by_variant_domain": {
            variant: dict(sorted(counter.items()))
            for variant, counter in sorted(by_variant_domain.items())
        },
        "agent_failure_buckets": dict(sorted(agent_failure_buckets.items())),
        "agent_gain_loss": dict(sorted(agent_gain_loss.items())),
        "agent_selection_methods": dict(sorted(agent_selection_buckets.items())),
        "verified_or_abstain_gate_status": dict(sorted(verified_gate_buckets.items())),
        "raw_content_persisted": False,
    }


def _first_agent_row(by_variant: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for variant, row in sorted(by_variant.items()):
        if str(variant).startswith("assumption_agent"):
            return row
    return None


def _diagnostic_domain(row: dict[str, Any]) -> str:
    text = " ".join([
        str(row.get("category") or ""),
        str(row.get("raw_subject") or ""),
    ]).lower()
    if any(token in text for token in ("math", "algebra", "geometry", "number theory", "combinatorics")):
        return "math"
    if any(token in text for token in ("physics", "chemistry", "biology", "medicine", "science")):
        return "science"
    if any(token in text for token in ("computer", "software", "program", "code", "algorithm")):
        return "software_engineering"
    if any(token in text for token in ("philosophy", "history", "law", "literature", "social")):
        return "humanities_social_science"
    return "hle_general"


def _row_flags(row: dict[str, Any]) -> dict[str, Any]:
    efficacy = row.get("component_efficacy")
    if not isinstance(efficacy, dict):
        return {}
    flags = efficacy.get("flags")
    return flags if isinstance(flags, dict) else {}


def _verified_gate_status(row: dict[str, Any]) -> str:
    efficacy = row.get("component_efficacy")
    if not isinstance(efficacy, dict):
        return "unknown"
    selection = efficacy.get("selection")
    if not isinstance(selection, dict):
        return "unknown"
    gate = selection.get("verified_or_abstain_gate")
    if not isinstance(gate, dict):
        return "not_recorded"
    return str(gate.get("status") or "unknown")


def _selection_is_verifier_like(method: str) -> bool:
    return method in {
        "verifier_choice",
        "source_grounded_verifier_choice",
        "counter_assumption_verifier_choice",
        "option_evidence_verifier_choice",
        "verifier_fallback_first",
        "counter_assumption_verifier_fallback_first",
        "counter_assumption_verifier_error_fallback_majority",
    }


def _jsonl_error_events(specs: list[ShardSpec]) -> dict[str, Counter[str]]:
    by_event: Counter[str] = Counter()
    by_variant: Counter[str] = Counter()
    by_error_type: Counter[str] = Counter()
    by_error_label: Counter[str] = Counter()
    for spec in specs:
        if not spec.log_out.exists():
            continue
        try:
            with spec.log_out.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    name = str(event.get("event") or "")
                    if name not in ERROR_EVENT_NAMES:
                        continue
                    by_event[name] += 1
                    by_variant[str(event.get("variant") or "unknown")] += 1
                    by_error_type[str(event.get("error_type") or "unknown")] += 1
                    by_error_label[_sanitize_error_label(event.get("error") or event.get("error_type") or "unknown")] += 1
        except OSError:
            continue
    return {
        "by_event": by_event,
        "by_variant": by_variant,
        "by_error_type": by_error_type,
        "by_error_label": by_error_label,
    }


def _sanitize_error_label(value: Any) -> str:
    text = " ".join(str(value or "unknown").split())
    if not text:
        return "unknown"
    # Keep endpoint/runtime labels, not prompt content or long provider payloads.
    replacements = {
        "Remote end closed connection without response": "RemoteDisconnected",
        "The read operation timed out": "ReadTimeout",
        "timed out": "Timeout",
    }
    for needle, label in replacements.items():
        if needle.lower() in text.lower():
            return label
    if len(text) > 120:
        return text[:117] + "..."
    return text


def format_parallel_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    errors = payload["error_stratification"]
    pollution = payload.get("pollution_audit") or {}
    model_budget = payload.get("model_budget_fairness_audit") or {}
    diagnostics = payload.get("failure_diagnostics") or {}
    operator_activation = metrics.get("operator_activation_summary") or {}
    operator_application = metrics.get("operator_application_summary") or {}
    claim_guard = pollution.get("claim_guard") or {}
    runtime_policy = payload.get("runtime_policy") or {}
    sampling = payload.get("sampling") or {}
    shard_dedupe = (payload.get("sampling") or {}).get("shard_sample_dedupe") or {"enabled": False}
    reuse_summary = runtime_policy.get("reuse_completed_shards") or {"enabled": False}
    lines = [
        "# HLE Parallel Shard Evaluation",
        "",
        f"- pass: `{payload['pass']}`",
        f"- paper clean pass: `{payload['paper_clean_pass']}`",
        f"- pollution pass: `{payload.get('pollution_pass')}`",
        f"- parallel workers: `{sampling.get('parallel_workers')}`",
        f"- launch stagger sec: `{runtime_policy.get('launch_stagger_sec')}`",
        f"- process timeout policy: `{runtime_policy.get('process_timeout_policy')}`",
        f"- kill on soft timeout: `{runtime_policy.get('kill_on_soft_timeout')}`",
        f"- reused completed shards: `{reuse_summary.get('reused_shard_count', 0)}`",
        f"- shard sample dedupe: `{shard_dedupe.get('status', shard_dedupe.get('enabled'))}`",
        f"- loaded shard payloads: `{payload['loaded_shard_payload_count']}/{payload['sampling']['planned_shard_count']}`",
        f"- sample count: `{metrics['sample_count']}`",
        f"- distinct sample problems: `{metrics['distinct_sample_problem_count']}`",
        f"- duplicate sample problems: `{metrics['duplicate_sample_problem_count']}`",
        f"- live attempts resolved: `{metrics['resolved_live_model_calls']}/{metrics['planned_live_model_calls']}`",
        f"- scored rows: `{metrics['scored_row_count']}`",
        f"- overall accuracy: `{metrics['overall_accuracy']}`",
        f"- top-level live errors: `{errors['top_level_error_count']}`",
        f"- process timeouts: `{errors['process_timeout_count']}`",
        f"- failed gates: `{payload['failed_gates']}`",
        f"- paper-clean failed gates: `{payload['paper_clean_failed_gates']}`",
        f"- pollution failed gates: `{payload.get('pollution_failed_gates')}`",
        f"- model-budget fairness failed gates: `{model_budget.get('failed_gates')}`",
        f"- operator selected/activated/blocked rows: "
        f"`{operator_activation.get('selected_row_count', 0)}/"
        f"{operator_activation.get('activated_row_count', 0)}/"
        f"{operator_activation.get('blocked_row_count', 0)}`",
        f"- operator status counts: `{operator_activation.get('status_counts', {})}`",
        f"- operator reason counts: `{operator_activation.get('reason_counts', {})}`",
        f"- operator application verifier rows: `{operator_application.get('verifier_activated_count', 0)}`",
        f"- operator average slot completion: `{operator_application.get('average_slot_completion_rate')}`",
        f"- operator changed-candidate rows: `{operator_application.get('changed_candidate_count', 0)}`",
        f"- operator decorative-use rate: `{operator_application.get('decorative_use_rate', 0.0)}`",
        f"- recommended HLE claim scope: `{claim_guard.get('recommended_hle_claim_scope')}`",
        "",
        "## By Variant",
        "",
        "| model | variant | n | accuracy | error count | MCQ accuracy | exact accuracy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, row in sorted(metrics["by_model_variant"].items()):
        model, variant = key.split("::", 1)
        lines.append(
            f"| `{model}` | `{variant}` | `{row['n']}` | `{row['accuracy']}` | "
            f"`{row['error_count']}` | `{row['multiple_choice_accuracy']}` | "
            f"`{row['exact_match_accuracy']}` |"
        )
    lines.extend([
        "",
        "## Clean Shared Subset",
        "",
        "| model | variant | clean shared n | accuracy |",
        "| --- | --- | ---: | ---: |",
    ])
    for model, row in sorted(metrics.get("clean_shared_subset", {}).items()):
        for variant, variant_row in sorted(row.get("by_variant", {}).items()):
            lines.append(
                f"| `{model}` | `{variant}` | `{variant_row['n']}` | `{variant_row['accuracy']}` |"
            )
    lines.extend([
        "",
        "## Route Credit",
        "",
        "| model | problems | agent acc | recoverable agent errors | unrecoverable agent errors | losses to controls | VOI actions |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    route_credit = metrics.get("route_credit_table", {})
    for model, row in sorted((route_credit.get("by_model") or {}).items()):
        losses = ", ".join(
            f"{variant}:{count}" for variant, count in sorted(row.get("agent_loss_to_control_counts", {}).items())
        ) or "none"
        voi_actions = ", ".join(
            f"{action}:{count}" for action, count in sorted(row.get("voi_recommended_action_counts", {}).items())
        ) or "none"
        lines.append(
            f"| `{model}` | `{row['problem_count']}` | `{row['agent_accuracy']}` | "
            f"`{row['recoverable_agent_error_count']}` | `{row['unrecoverable_agent_error_count']}` | "
            f"`{losses}` | `{voi_actions}` |"
        )
    lines.extend([
        "",
        "## Error Stratification",
        "",
        "| bucket | key | count |",
        "| --- | --- | ---: |",
    ])
    for bucket in (
        "top_level_errors_by_variant",
        "top_level_errors_by_type",
        "top_level_errors_by_variant_type",
        "top_level_errors_by_label",
        "top_level_errors_by_variant_label",
        "jsonl_error_events_by_event",
        "jsonl_error_events_by_variant",
        "jsonl_error_events_by_type",
        "jsonl_error_events_by_label",
        "process_status_counts",
    ):
        for key, count in sorted((errors.get(bucket) or {}).items()):
            lines.append(f"| `{bucket}` | `{key}` | `{count}` |")
    context_summary = (pollution.get("context_pollution") or {}).get("summary") or {}
    lines.extend([
        "",
        "## Pollution Audit",
        "",
        "| bucket | key | value |",
        "| --- | --- | ---: |",
    ])
    for key, value in sorted((pollution.get("fresh_problem_hash_exclusion") or {}).items()):
        lines.append(f"| `fresh_problem_hash_exclusion` | `{key}` | `{value}` |")
    for key, value in sorted((pollution.get("cache_live_separation") or {}).items()):
        if isinstance(value, dict):
            continue
        lines.append(f"| `cache_live_separation` | `{key}` | `{value}` |")
    for key, value in sorted(context_summary.items()):
        lines.append(f"| `context_pollution_summary` | `{key}` | `{value}` |")
    for key, value in sorted((pollution.get("gates") or {}).items()):
        lines.append(f"| `pollution_gate` | `{key}` | `{value}` |")
    lines.extend([
        "",
        "## Model Budget Fairness",
        "",
        "| bucket | key | value |",
        "| --- | --- | --- |",
    ])
    for key in (
        "agent_row_count",
        "agent_top_models",
        "stronger_or_different_effective_models",
        "multi_call_agent_row_count",
        "budget_target_models",
        "missing_same_model_controls",
        "missing_strong_baseline_controls",
        "missing_budget_matched_controls",
    ):
        lines.append(f"| `summary` | `{key}` | `{model_budget.get(key)}` |")
    for key, value in sorted((model_budget.get("gates") or {}).items()):
        lines.append(f"| `fairness_gate` | `{key}` | `{value}` |")
    lines.extend([
        "",
        "## Selection Credit",
        "",
        "| method | n | correct | error | accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for method, row in sorted(((pollution.get("module_credit_assignment") or {}).get("by_selection_method") or {}).items()):
        lines.append(
            f"| `{method}` | `{row['n']}` | `{row['correct']}` | `{row['error']}` | `{row['accuracy']}` |"
        )
    lines.extend([
        "",
        "## Failure Diagnostics",
        "",
        "| bucket | key | count |",
        "| --- | --- | ---: |",
    ])
    for bucket in (
        "agent_failure_buckets",
        "agent_gain_loss",
        "agent_selection_methods",
        "verified_or_abstain_gate_status",
    ):
        for key, value in sorted((diagnostics.get(bucket) or {}).items()):
            lines.append(f"| `{bucket}` | `{key}` | `{value}` |")
    for variant, counts in sorted((diagnostics.get("by_variant_answer_type") or {}).items()):
        for key, value in sorted(counts.items()):
            lines.append(f"| `by_variant_answer_type::{variant}` | `{key}` | `{value}` |")
    for variant, counts in sorted((diagnostics.get("by_variant_domain") or {}).items()):
        for key, value in sorted(counts.items()):
            lines.append(f"| `by_variant_domain::{variant}` | `{key}` | `{value}` |")
    lines.extend([
        "",
        "## Shards",
        "",
        "| shard | status | returncode | elapsed sec | sample size | seed offset | timeout |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for shard in sorted(payload.get("shards", []), key=lambda item: item["shard_index"]):
        timeout = (
            "hard"
            if shard.get("hard_kill_sent")
            else "soft-kill"
            if shard.get("soft_timeout_sent")
            else "soft-observed"
            if shard.get("soft_timeout_observed")
            else "none"
        )
        lines.append(
            f"| `{shard['shard_index']}` | `{shard['status']}` | `{shard['returncode']}` | "
            f"`{shard['elapsed_sec']}` | `{shard['sample_size']}` | `{shard['seed_offset']}` | `{timeout}` |"
        )
    lines.extend([
        "",
        "Raw HLE questions, answers, rationales, canaries, and prediction text are not persisted.",
    ])
    return "\n".join(lines) + "\n"


def build_payload_without_execution(args: argparse.Namespace) -> tuple[list[ShardSpec], list[ShardRunState]]:
    root = Path(args.root).resolve()
    run_dir = _path_arg(args.run_dir, root=root)
    md_dir = _path_arg(args.md_dir, root=root)
    explicit_seed_offsets = _parse_seed_offsets(getattr(args, "seed_offsets", ""))
    if explicit_seed_offsets:
        if args.shard_size != 1:
            raise ValueError("--seed-offsets requires --shard-size 1")
        specs = build_shard_specs_for_seed_offsets(
            eval_id=args.eval_id,
            seed_offsets=explicit_seed_offsets,
            run_dir=run_dir,
            md_dir=md_dir,
        )
    else:
        specs = build_shard_specs(
            eval_id=args.eval_id,
            total_sample_size=args.total_sample_size,
            shard_size=args.shard_size,
            seed_offset=args.seed_offset,
            seed_stride=args.seed_stride,
            run_dir=run_dir,
            md_dir=md_dir,
        )
    dedupe_summary: dict[str, Any] = {"enabled": False}
    if explicit_seed_offsets:
        dedupe_summary = {
            "enabled": False,
            "reason": "explicit_seed_offsets",
            "raw_content_persisted": False,
            "distinct_problem_hash_count": None,
            "seed_offsets": explicit_seed_offsets,
        }
    elif getattr(args, "dedupe_shard_samples", False):
        try:
            specs, dedupe_summary = dedupe_shard_specs_by_sample_hash(
                root=root,
                specs=specs,
                max_scan=args.max_scan,
                seed_stride=args.seed_stride,
                exclude_existing_hle_artifacts=args.exclude_existing_hle_artifacts,
                exclude_artifact_glob=args.exclude_artifact_glob,
                sample_answer_type=args.sample_answer_type,
                sample_subject_contains=args.sample_subject_contains,
                max_attempts=args.dedupe_shard_max_attempts,
            )
        except Exception as exc:
            dedupe_summary = {
                "enabled": True,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc)[:240],
                "raw_content_persisted": False,
            }
    setattr(args, "_shard_sample_dedupe_summary", dedupe_summary)
    graph_dir = _path_arg(args.graph_dir, root=root)
    states = [
        ShardRunState(
            spec=spec,
            command=build_shard_command(
                spec,
                root=root,
                max_scan=args.max_scan,
                models=args.models,
                variants=args.variants,
                execute_live=args.execute_live,
                call_timeout=args.call_timeout,
                max_tokens=args.max_tokens,
                graph_dir=graph_dir,
                agent_top_k=args.agent_top_k,
                agent_context_max_chars=args.agent_context_max_chars,
                agent_child_mode=args.agent_child_mode,
                agent_child_timeout=args.agent_child_timeout,
                evidence_bridge_enabled=not args.disable_evidence_bridge,
                exclude_existing_hle_artifacts=args.exclude_existing_hle_artifacts,
                exclude_artifact_glob=args.exclude_artifact_glob,
                sample_answer_type=args.sample_answer_type,
                sample_subject_contains=args.sample_subject_contains,
                enable_assumption_operators=bool(getattr(args, "enable_assumption_operators", False)),
                disable_assumption_operators=bool(getattr(args, "disable_assumption_operators", False)),
                assumption_operator_domains=str(getattr(args, "assumption_operator_domains", "") or ""),
                assumption_operator_skip_domains=str(getattr(args, "assumption_operator_skip_domains", "") or ""),
                assumption_operator_max_specs=getattr(args, "assumption_operator_max_specs", None),
                allow_assumption_operators_without_context=bool(
                    getattr(args, "allow_assumption_operators_without_context", False)
                ),
                enable_assumption_operator_retrieval_fallback=bool(
                    getattr(args, "enable_assumption_operator_retrieval_fallback", False)
                ),
                assumption_operator_fallback_min_score=getattr(
                    args, "assumption_operator_fallback_min_score", None
                ),
                enable_operator_application_verifier=bool(
                    getattr(args, "enable_operator_application_verifier", False)
                ),
                disable_domain_rule_verifier=bool(getattr(args, "disable_domain_rule_verifier", False)),
                enable_option_claim_contrastive_adjudicator=bool(
                    getattr(args, "enable_option_claim_contrastive_adjudicator", False)
                ),
                disable_option_claim_contrastive_adjudicator=bool(
                    getattr(args, "disable_option_claim_contrastive_adjudicator", False)
                ),
                enable_option_claim_span_directness_verifier=bool(
                    getattr(args, "enable_option_claim_span_directness_verifier", False)
                ),
                disable_option_claim_span_directness_verifier=bool(
                    getattr(args, "disable_option_claim_span_directness_verifier", False)
                ),
            ),
        )
        for spec in specs
    ]
    return specs, states


def _path_arg(value: str, *, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HLE smoke eval through parallel shards.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="hle_parallel_shard_eval_20260616")
    parser.add_argument("--total-sample-size", type=int, default=30)
    parser.add_argument("--shard-size", type=int, default=1)
    parser.add_argument("--parallel-workers", type=int, default=3)
    parser.add_argument("--max-scan", type=int, default=5000)
    parser.add_argument("--seed-offset", type=int, default=3000)
    parser.add_argument("--seed-stride", type=int, default=400)
    parser.add_argument(
        "--seed-offsets",
        default="",
        help="Comma-separated explicit seed offsets. Requires --shard-size 1 and skips parent dedupe remapping.",
    )
    parser.add_argument("--dedupe-shard-samples", action="store_true")
    parser.add_argument("--dedupe-shard-max-attempts", type=int, default=25)
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
    parser.add_argument("--enable-assumption-operators", action="store_true")
    parser.add_argument("--disable-assumption-operators", action="store_true")
    parser.add_argument("--assumption-operator-domains", default="")
    parser.add_argument("--assumption-operator-skip-domains", default="")
    parser.add_argument("--assumption-operator-max-specs", type=int, default=None)
    parser.add_argument("--allow-assumption-operators-without-context", action="store_true")
    parser.add_argument("--enable-assumption-operator-retrieval-fallback", action="store_true")
    parser.add_argument("--assumption-operator-fallback-min-score", type=float, default=None)
    parser.add_argument("--enable-operator-application-verifier", action="store_true")
    parser.add_argument("--disable-domain-rule-verifier", action="store_true")
    parser.add_argument("--enable-option-claim-contrastive-adjudicator", action="store_true")
    parser.add_argument("--disable-option-claim-contrastive-adjudicator", action="store_true")
    parser.add_argument("--enable-option-claim-span-directness-verifier", action="store_true")
    parser.add_argument("--disable-option-claim-span-directness-verifier", action="store_true")
    parser.add_argument("--exclude-existing-hle-artifacts", action="store_true")
    parser.add_argument(
        "--exclude-artifact-glob",
        default="phase four/assumption_graph/paper_readiness_20260604/hle_parallel_runs/hle*.json*",
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--md-dir", default=str(DEFAULT_MD_DIR))
    parser.add_argument("--out", default="")
    parser.add_argument("--md-out", default="")
    parser.add_argument("--heartbeat-out", default="")
    parser.add_argument(
        "--log-out",
        default="",
        help="Metadata-only parent-runner JSONL diagnostic log. Defaults to <run-dir>/<eval-id>.diagnostic.jsonl.",
    )
    parser.add_argument("--poll-interval-sec", type=float, default=2.0)
    parser.add_argument("--heartbeat-interval-sec", type=float, default=10.0)
    parser.add_argument("--launch-stagger-sec", type=float, default=0.0)
    parser.add_argument("--reuse-completed-shards", action="store_true")
    parser.add_argument("--soft-timeout-sec", type=float, default=None)
    parser.add_argument("--terminate-grace-sec", type=float, default=30.0)
    parser.add_argument(
        "--kill-on-soft-timeout",
        action="store_true",
        help="Terminate and then kill shards after --soft-timeout-sec. Default only records heartbeat observation.",
    )
    parser.add_argument("--model-router-attempts", type=int, default=None)
    parser.add_argument("--model-router-timeout", type=float, default=None)
    parser.add_argument("--model-router-per-attempt-timeout", type=float, default=None)
    parser.add_argument("--model-router-subprocess-calls", action="store_true")
    parser.add_argument("--model-router-no-byte-timeout-sec", type=float, default=None)
    parser.add_argument("--model-router-backoff-base-sec", type=float, default=None)
    parser.add_argument("--model-router-global-concurrency", type=int, default=None)
    parser.add_argument("--model-router-global-concurrency-dir", default="")
    parser.add_argument("--model-router-global-slot-ttl-sec", type=float, default=None)
    parser.add_argument("--model-router-global-slot-wait-sec", type=float, default=None)
    parser.add_argument("--skip-live-model-preflight", action="store_true")
    parser.add_argument("--live-model-preflight-timeout-sec", type=float, default=60.0)
    args = parser.parse_args()
    args = apply_live_network_defaults(args)
    apply_hle_offline_defaults(os.environ)

    root = Path(args.root).resolve()
    run_dir = _path_arg(args.run_dir, root=root)
    out = _path_arg(args.out, root=root) if args.out else run_dir / f"{args.eval_id}.json"
    md_out = _path_arg(args.md_out, root=root) if args.md_out else _path_arg(args.md_dir, root=root) / f"{args.eval_id}.md"
    heartbeat_path = (
        _path_arg(args.heartbeat_out, root=root)
        if args.heartbeat_out
        else run_dir / f"{args.eval_id}.heartbeat.json"
    )
    diagnostic_log_out = (
        _path_arg(args.log_out, root=root)
        if args.log_out
        else run_dir / f"{args.eval_id}.diagnostic.jsonl"
    )
    logger = JsonlDiagnosticLogger(diagnostic_log_out)
    log_event(
        logger,
        {
            "event": "hle_parallel_runner_started",
            "eval_id": args.eval_id,
            "run_dir": str(run_dir),
            "out": str(out),
            "md_out": str(md_out),
            "heartbeat_out": str(heartbeat_path),
            "total_sample_size": int(args.total_sample_size),
            "shard_size": int(args.shard_size),
            "parallel_workers": int(args.parallel_workers),
            "models": [item.strip() for item in str(args.models).split(",") if item.strip()],
            "variants": [item.strip() for item in str(args.variants).split(",") if item.strip()],
            "execute_live": bool(args.execute_live),
            "soft_timeout_sec": args.soft_timeout_sec,
            "kill_on_soft_timeout": bool(args.kill_on_soft_timeout),
            "model_router": {
                "attempts": args.model_router_attempts,
                "timeout_sec": args.model_router_timeout,
                "per_attempt_timeout_sec": args.model_router_per_attempt_timeout,
                "subprocess_calls": bool(args.model_router_subprocess_calls),
                "subprocess_no_byte_timeout_sec": args.model_router_no_byte_timeout_sec,
                "global_concurrency": args.model_router_global_concurrency,
                "global_slot_ttl_sec": args.model_router_global_slot_ttl_sec,
                "global_slot_wait_sec": args.model_router_global_slot_wait_sec,
                "raw_content_persisted": False,
            },
            "reuse_completed_shards": bool(args.reuse_completed_shards),
            "dedupe_shard_samples": bool(args.dedupe_shard_samples),
        },
    )
    hash_cache_path = run_dir / f"{args.eval_id}.existing_hash_cache.json"
    os.environ.setdefault("HLE_EXISTING_HASH_CACHE_PATH", str(hash_cache_path))
    write_preflight_heartbeat(
        heartbeat_path,
        eval_id=args.eval_id,
        phase="building_shards",
        run_dir=run_dir,
        details={
            "total_sample_size": args.total_sample_size,
            "shard_size": args.shard_size,
            "dedupe_shard_samples": bool(args.dedupe_shard_samples),
            "exclude_existing_hle_artifacts": bool(args.exclude_existing_hle_artifacts),
            "hash_cache_path": str(hash_cache_path),
        },
    )
    specs, states = build_payload_without_execution(args)
    log_event(
        logger,
        {
            "event": "hle_parallel_runner_shards_built",
            "eval_id": args.eval_id,
            "shard_count": len(states),
            "seed_offsets": [state.spec.seed_offset for state in states],
            "sample_sizes": [state.spec.sample_size for state in states],
            "dedupe_summary": getattr(args, "_shard_sample_dedupe_summary", {"enabled": False}),
            "hash_cache_path": str(hash_cache_path),
        },
    )
    write_preflight_heartbeat(
        heartbeat_path,
        eval_id=args.eval_id,
        phase="shards_built",
        run_dir=run_dir,
        details={
            "shard_count": len(states),
            "dedupe_summary": getattr(args, "_shard_sample_dedupe_summary", {"enabled": False}),
            "hash_cache_path": str(hash_cache_path),
        },
    )
    reuse_summary = (
        mark_reusable_completed_shards(states)
        if args.reuse_completed_shards
        else {"enabled": False}
    )
    log_event(
        logger,
        {
            "event": "hle_parallel_runner_reuse_summary",
            "eval_id": args.eval_id,
            "reuse_summary": reuse_summary,
            "reused_shard_count": sum(1 for state in states if state.reused_existing_payload),
        },
    )
    env = build_runner_env(
        model_router_attempts=args.model_router_attempts,
        model_router_timeout=args.model_router_timeout,
        parallel_workers=args.parallel_workers,
        model_router_per_attempt_timeout=args.model_router_per_attempt_timeout,
        model_router_subprocess_calls=True if args.model_router_subprocess_calls else None,
        model_router_no_byte_timeout_sec=args.model_router_no_byte_timeout_sec,
        model_router_backoff_base_sec=args.model_router_backoff_base_sec,
        model_router_global_concurrency=args.model_router_global_concurrency,
        model_router_global_concurrency_dir=args.model_router_global_concurrency_dir,
        model_router_global_slot_ttl_sec=args.model_router_global_slot_ttl_sec,
        model_router_global_slot_wait_sec=args.model_router_global_slot_wait_sec,
    )
    if args.execute_live and not model_router_primary_key_present(env):
        model_preflight = run_live_model_preflight(
            models=args.models,
            env=env,
            timeout_sec=0.0,
        )
        log_event(
            logger,
            {
                "event": "hle_parallel_runner_model_key_missing",
                "eval_id": args.eval_id,
                "model_preflight": model_preflight,
                "skip_live_model_preflight": bool(args.skip_live_model_preflight),
            },
        )
        write_preflight_heartbeat(
            heartbeat_path,
            eval_id=args.eval_id,
            phase="model_key_missing",
            run_dir=run_dir,
            details=model_preflight,
        )
        print(json.dumps({
            "eval_id": args.eval_id,
            "pass": False,
            "failed_gates": ["live_model_primary_key_present"],
            "model_preflight": model_preflight,
            "heartbeat_out": str(heartbeat_path),
            "log_out": str(diagnostic_log_out),
            "raw_content_persisted": False,
        }, ensure_ascii=True, indent=2, sort_keys=True))
        raise SystemExit(2)
    if args.execute_live and not args.skip_live_model_preflight:
        model_preflight = run_live_model_preflight(
            models=args.models,
            env=env,
            timeout_sec=float(args.live_model_preflight_timeout_sec or 0.0),
        )
        if not model_preflight.get("passed"):
            log_event(
                logger,
                {
                    "event": "hle_parallel_runner_model_preflight_failed",
                    "eval_id": args.eval_id,
                    "model_preflight": model_preflight,
                },
            )
            write_preflight_heartbeat(
                heartbeat_path,
                eval_id=args.eval_id,
                phase="model_preflight_failed",
                run_dir=run_dir,
                details=model_preflight,
            )
            print(json.dumps({
                "eval_id": args.eval_id,
                "pass": False,
                "failed_gates": ["live_model_preflight"],
                "model_preflight": model_preflight,
                "heartbeat_out": str(heartbeat_path),
                "log_out": str(diagnostic_log_out),
                "raw_content_persisted": False,
            }, ensure_ascii=True, indent=2, sort_keys=True))
            raise SystemExit(2)
        log_event(
            logger,
            {
                "event": "hle_parallel_runner_model_preflight_passed",
                "eval_id": args.eval_id,
                "model_preflight": model_preflight,
            },
        )
    run_parallel_shards(
        root=root,
        shard_states=states,
        parallel_workers=args.parallel_workers,
        heartbeat_path=heartbeat_path,
        poll_interval_sec=args.poll_interval_sec,
        heartbeat_interval_sec=args.heartbeat_interval_sec,
        soft_timeout_sec=args.soft_timeout_sec,
        terminate_grace_sec=args.terminate_grace_sec,
        kill_on_soft_timeout=args.kill_on_soft_timeout,
        launch_stagger_sec=max(0.0, float(args.launch_stagger_sec or 0.0)),
        env=env,
    )
    log_event(
        logger,
        {
            "event": "hle_parallel_runner_shards_run_completed",
            "eval_id": args.eval_id,
            "shard_status_counts": dict(sorted(Counter(state.status for state in states).items())),
            "returncode_counts": dict(sorted(Counter(str(state.returncode) for state in states).items())),
            "soft_timeout_observed_count": sum(1 for state in states if state.soft_timeout_observed),
            "hard_kill_sent_count": sum(1 for state in states if state.hard_kill_sent),
            "elapsed_sec_by_shard": {
                str(state.spec.shard_index): state.elapsed_sec()
                for state in states
            },
            "process_peak_rss_kb_by_shard": {
                str(state.spec.shard_index): state.peak_rss_kb
                for state in states
            },
            "process_peak_vms_kb_by_shard": {
                str(state.spec.shard_index): state.peak_vms_kb
                for state in states
            },
        },
    )
    payloads = load_shard_payloads(specs)
    log_event(
        logger,
        {
            "event": "hle_parallel_runner_payloads_loaded",
            "eval_id": args.eval_id,
            "loaded_shard_payload_count": len(payloads),
            "expected_shard_payload_count": len(specs),
            "row_count": sum(len(payload.get("rows") or payload.get("run_rows") or []) for payload in payloads),
        },
    )
    payload = aggregate_parallel_payload(
        eval_id=args.eval_id,
        specs=specs,
        states=states,
        shard_payloads=payloads,
        execute_live=args.execute_live,
        models=args.models,
        variants=args.variants,
        total_sample_size=args.total_sample_size,
        shard_size=args.shard_size,
        parallel_workers=args.parallel_workers,
        soft_timeout_sec=args.soft_timeout_sec,
        kill_on_soft_timeout=args.kill_on_soft_timeout,
        shard_sample_dedupe=getattr(args, "_shard_sample_dedupe_summary", {"enabled": False}),
        reuse_completed_shards=reuse_summary,
        launch_stagger_sec=max(0.0, float(args.launch_stagger_sec or 0.0)),
        diagnostic_log_out=diagnostic_log_out,
        model_router_policy=model_router_policy_from_env(env),
    )
    log_event(
        logger,
        {
            "event": "hle_parallel_runner_aggregate_built",
            "eval_id": args.eval_id,
            "pass": bool(payload.get("pass")),
            "paper_clean_pass": bool(payload.get("paper_clean_pass")),
            "pollution_pass": bool(payload.get("pollution_pass")),
            "failed_gates": list(payload.get("failed_gates") or []),
            "paper_clean_failed_gates": list(payload.get("paper_clean_failed_gates") or []),
            "pollution_failed_gates": list(payload.get("pollution_failed_gates") or []),
            "metrics": {
                "sample_count": payload["metrics"]["sample_count"],
                "distinct_sample_problem_count": payload["metrics"]["distinct_sample_problem_count"],
                "duplicate_sample_problem_count": payload["metrics"]["duplicate_sample_problem_count"],
                "scored_row_count": payload["metrics"]["scored_row_count"],
                "overall_accuracy": payload["metrics"]["overall_accuracy"],
                "resolved_live_model_calls": payload["metrics"]["resolved_live_model_calls"],
                "planned_live_model_calls": payload["metrics"]["planned_live_model_calls"],
            },
            "model_budget_fairness_failed_gates": payload["model_budget_fairness_audit"]["failed_gates"],
            "failure_diagnostics": {
                "agent_failure_buckets": payload["failure_diagnostics"]["agent_failure_buckets"],
                "agent_gain_loss": payload["failure_diagnostics"]["agent_gain_loss"],
                "verified_or_abstain_gate_status": payload["failure_diagnostics"]["verified_or_abstain_gate_status"],
            },
        },
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(format_parallel_markdown(payload), encoding="utf-8")
    log_event(
        logger,
        {
            "event": "hle_parallel_runner_artifacts_written",
            "eval_id": args.eval_id,
            "out": str(out),
            "md_out": str(md_out),
            "heartbeat_out": str(heartbeat_path),
            "log_out": str(diagnostic_log_out),
        },
    )
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "paper_clean_pass": payload["paper_clean_pass"],
        "pollution_pass": payload["pollution_pass"],
        "metrics": {
            "sample_count": payload["metrics"]["sample_count"],
            "distinct_sample_problem_count": payload["metrics"]["distinct_sample_problem_count"],
            "duplicate_sample_problem_count": payload["metrics"]["duplicate_sample_problem_count"],
            "scored_row_count": payload["metrics"]["scored_row_count"],
            "overall_accuracy": payload["metrics"]["overall_accuracy"],
            "resolved_live_model_calls": payload["metrics"]["resolved_live_model_calls"],
            "planned_live_model_calls": payload["metrics"]["planned_live_model_calls"],
        },
        "error_stratification": {
            "top_level_error_count": payload["error_stratification"]["top_level_error_count"],
            "process_timeout_count": payload["error_stratification"]["process_timeout_count"],
        },
        "pollution_audit": {
            "recommended_hle_claim_scope": payload["pollution_audit"]["claim_guard"]["recommended_hle_claim_scope"],
            "failed_gates": payload["pollution_failed_gates"],
        },
        "model_budget_fairness_audit": {
            "failed_gates": payload["model_budget_fairness_audit"]["failed_gates"],
            "stronger_or_different_effective_models": (
                payload["model_budget_fairness_audit"]["stronger_or_different_effective_models"]
            ),
            "multi_call_agent_row_count": payload["model_budget_fairness_audit"]["multi_call_agent_row_count"],
        },
        "failure_diagnostics": {
            "agent_failure_buckets": payload["failure_diagnostics"]["agent_failure_buckets"],
            "agent_gain_loss": payload["failure_diagnostics"]["agent_gain_loss"],
            "verified_or_abstain_gate_status": payload["failure_diagnostics"]["verified_or_abstain_gate_status"],
        },
        "failed_gates": payload["failed_gates"],
        "paper_clean_failed_gates": payload["paper_clean_failed_gates"],
        "out": str(out),
        "heartbeat_out": str(heartbeat_path),
        "log_out": str(diagnostic_log_out),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
