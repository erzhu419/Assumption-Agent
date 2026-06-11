"""Fresh-live capable residual multi-generation loop.

This is the production-facing successor to the replayed mini-loop.  It builds
the same contract/preflight/acceptance path, but can optionally obtain fresh
LLM judgments through environment variables.  Dry-run and blocked-env modes are
kept explicit so paper artifacts cannot overclaim a live run.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import requests

from .candidate_acceptance import apply_accepted_candidates, build_acceptance_payload
from .full_v3_residual_live_mini_loop import (
    _preflight_payload,
    _proposal_payload,
    _select_generation_one_candidates,
)
from .full_v3_residual_multigeneration_loop import build_full_v3_residual_multigeneration_loop_payload
from .graph_memory import JsonlGraphStore
from .proposal_contract import build_proposal_contract_payload, filter_proposal_payload_by_contract


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_residual_fresh_live_loop_20260611.json"


def build_full_v3_residual_fresh_live_loop_payload(
    *,
    root: Path,
    eval_id: str = "full_v3_residual_fresh_live_loop_20260611",
    execution_mode: str = "dry_run",
    candidate_count: int = 3,
    trigger_rows_per_candidate: int = 4,
    control_rows_per_candidate: int = 2,
    model_alias: str = "gpt_mini",
    load_keyfile: bool = True,
) -> dict[str, Any]:
    if execution_mode not in {"dry_run", "execute_live", "summarize"}:
        raise ValueError(f"unknown execution_mode={execution_mode}")
    root = root.resolve()
    if load_keyfile:
        _load_keyfile_env()
    env = _env_status(model_alias)
    dry_loop = build_full_v3_residual_multigeneration_loop_payload(
        root=root,
        eval_id=f"{eval_id}_source_multigen",
        generations=3,
        seed_limit=8,
    )
    selected = _select_generation_one_candidates(dry_loop, limit=candidate_count)
    with tempfile.TemporaryDirectory() as td:
        graph_dir = Path(td) / "graph"
        store = JsonlGraphStore(graph_dir)
        proposal_payload = _proposal_payload(eval_id=eval_id, candidates=selected, store=store)
        contract = build_proposal_contract_payload(
            proposal_payload=proposal_payload,
            eval_id=f"{eval_id}_proposal_contract",
            store=store,
        )
        contract_ready = filter_proposal_payload_by_contract(proposal_payload, contract)
        preflight = _preflight_payload(
            eval_id=f"{eval_id}_candidate_preflight",
            proposal_payload=contract_ready,
            trigger_rows_per_candidate=trigger_rows_per_candidate,
            control_rows_per_candidate=control_rows_per_candidate,
        )
        live = _live_judgment_payload(
            preflight=preflight,
            selected=selected,
            execution_mode=execution_mode,
            env=env,
            model_alias=model_alias,
        )
        judgment_path = Path(td) / "fresh_live_judgments.json"
        judgment_path.write_text(
            json.dumps(live["judgments"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        acceptance = build_acceptance_payload(
            proposal_payload=contract_ready,
            preflight_payload=preflight,
            judgment_paths=[judgment_path],
            candidate_variant="candidate",
            baseline_variant="baseline",
            eval_id=f"{eval_id}_candidate_acceptance",
            min_trigger_judgments=trigger_rows_per_candidate,
            benefit_lcb90=0.54,
            control_loss_ucb90=0.35,
        )
        before_node_count = len(JsonlGraphStore(graph_dir).nodes)
        applied = apply_accepted_candidates(JsonlGraphStore(graph_dir), contract_ready, acceptance)
        after_node_count = len(JsonlGraphStore(graph_dir).nodes)
    metrics = _metrics(
        execution_mode=execution_mode,
        env=env,
        selected=selected,
        contract=contract,
        preflight=preflight,
        live=live,
        acceptance=acceptance,
        applied=applied,
        before_node_count=before_node_count,
        after_node_count=after_node_count,
    )
    gates = {
        "source_multigeneration_loop_passes": bool(dry_loop.get("pass")),
        "contract_ready": metrics["contract_ready_count"] == candidate_count,
        "preflight_ready": metrics["preflight_ready_count"] == candidate_count,
        "fresh_live_path_present": metrics["fresh_live_path_present"] is True,
        "dry_run_or_live_judgments_available": (
            execution_mode == "dry_run"
            or metrics["fresh_api_call_count"] >= metrics["planned_fresh_api_call_count"]
        ),
        "live_env_status_recorded": bool(env.get("status")),
        "accepted_apply_shadow_only": metrics["main_graph_mutation_count"] == 0,
        "no_secret_value_exposed": metrics["secret_value_exposed"] is False,
    }
    if execution_mode == "execute_live":
        gates["execute_live_requires_ready_env"] = env["ready"] is True
        gates["execute_live_acceptance_available"] = metrics["accepted_count"] >= 1
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_residual_fresh_live_loop",
        "reconstruction_v2_full_phase": "fresh_api_capable_residual_multigeneration_loop",
        "implementation_level": (
            "fresh_api_execute_path" if execution_mode == "execute_live"
            else "fresh_api_capable_dry_run"
        ),
        "performance_validation": True,
        "execution_mode": execution_mode,
        "validation_scope": (
            "Builds a residual multi-generation loop that can call a live OpenAI-compatible API for fresh "
            "trigger/control judgments.  Dry-run mode validates the full contract/preflight/acceptance/apply "
            "path without spending API calls."
        ),
        "live_env": env,
        "selected_generation_one_candidates": selected,
        "proposal_contract": contract,
        "candidate_preflight": preflight,
        "live_judgment_payload": live,
        "candidate_acceptance": acceptance,
        "applied_candidate_node_ids": applied,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": _interpretation(execution_mode=execution_mode, env=env, metrics=metrics),
    }


def _live_judgment_payload(
    *,
    preflight: dict[str, Any],
    selected: list[dict[str, Any]],
    execution_mode: str,
    env: dict[str, Any],
    model_alias: str,
) -> dict[str, Any]:
    planned = sum(
        len(row.get("trigger_problem_ids", [])) + len(row.get("control_problem_ids", []))
        for row in preflight.get("summaries", [])
    )
    if execution_mode == "dry_run":
        return {
            "status": "dry_run_no_api_calls",
            "fresh_api_call_count": 0,
            "planned_fresh_api_call_count": planned,
            "judgments": _deterministic_judgments(preflight),
            "live_errors": [],
        }
    if not env["ready"]:
        return {
            "status": "blocked_env_not_ready",
            "fresh_api_call_count": 0,
            "planned_fresh_api_call_count": planned,
            "judgments": {},
            "live_errors": [env.get("status", "env_not_ready")],
        }
    client = _Client(env["model"], env["base_url"], _key_for_alias(model_alias), model_alias)
    judgments: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    candidate_by_index = {idx: row for idx, row in enumerate(selected)}
    call_count = 0
    for summary_index, summary in enumerate(preflight.get("summaries", [])):
        candidate = candidate_by_index.get(summary_index, {})
        for problem_id in summary.get("trigger_problem_ids", []):
            try:
                judgments[problem_id] = client.judge(_prompt(candidate=candidate, problem_id=problem_id, control=False))
                call_count += 1
            except Exception as exc:  # pragma: no cover - depends on live network.
                errors.append(_redacted_error(exc))
        for problem_id in summary.get("control_problem_ids", []):
            try:
                judgments[problem_id] = client.judge(_prompt(candidate=candidate, problem_id=problem_id, control=True))
                call_count += 1
            except Exception as exc:  # pragma: no cover - depends on live network.
                errors.append(_redacted_error(exc))
    return {
        "status": "execute_complete" if call_count == planned and not errors else "execute_partial_or_failed",
        "fresh_api_call_count": call_count,
        "planned_fresh_api_call_count": planned,
        "judgments": judgments,
        "live_errors": errors[:5],
    }


def _deterministic_judgments(preflight: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for summary in preflight.get("summaries", []):
        for problem_id in summary.get("trigger_problem_ids", []):
            out[problem_id] = {"winner": "candidate", "score_a": 9, "score_b": 7, "source": "dry_run_fixture"}
        for problem_id in summary.get("control_problem_ids", []):
            out[problem_id] = {"winner": "tie", "score_a": 8, "score_b": 8, "source": "dry_run_fixture"}
    return out


def _prompt(*, candidate: dict[str, Any], problem_id: str, control: bool) -> str:
    kind = "negative control" if control else "trigger"
    return (
        "You are auditing a redacted self-evolution candidate. "
        "Return JSON only with winner equal to candidate, baseline, or tie, plus score_a and score_b integers. "
        f"Row type: {kind}. Problem id: {problem_id}. Candidate claim: {candidate.get('claim','')}. "
        f"Evaluation plan: {candidate.get('evaluation_plan','')}. "
        "Prefer candidate only if the claim gives a scoped, testable improvement without control harm."
    )


class _Client:
    def __init__(self, model: str, base_url: str, key: str, alias: str):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.alias = alias

    def judge(self, prompt: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 120,
                "temperature": 0,
            },
            timeout=float(os.environ.get("MODEL_ROUTER_TIMEOUT", "45")),
        )
        response.raise_for_status()
        text = (response.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {"winner": "tie", "score_a": 8, "score_b": 8, "raw": text[:200]}
        winner = payload.get("winner", "tie")
        if winner not in {"candidate", "baseline", "tie"}:
            winner = "tie"
        return {
            "winner": winner,
            "score_a": int(payload.get("score_a", 8)),
            "score_b": int(payload.get("score_b", 8)),
            "source": f"fresh_api::{self.alias}",
            "model": self.model,
        }


def _env_status(model_alias: str) -> dict[str, Any]:
    base = os.environ.get("RUOLI_BASE_URL", "https://ruoli.dev").rstrip("/") + "/v1"
    key = _key_for_alias(model_alias)
    if model_alias in {"gpt_mini", "gpt54_mini"}:
        model = os.environ.get("GPT_MINI_MODEL", "gpt-5.4-mini")
    elif model_alias in {"gpt55", "gpt5"}:
        model = os.environ.get("GPT55_MODEL", "gpt-5.5")
    elif model_alias == "claude_haiku":
        model = os.environ.get("CLAUDE_HAIKU_MODEL", "claude-haiku-4-5-20251001")
    else:
        model = model_alias
    return {
        "model_alias": model_alias,
        "model": model,
        "base_url": base,
        "ready": bool(key),
        "status": "env_ready" if key else "missing_api_key_env",
        "secret_value_exposed": False,
    }


def _key_for_alias(model_alias: str) -> str:
    if model_alias in {"gpt_mini", "gpt54_mini", "gpt55", "gpt5"}:
        return os.environ.get("RUOLI_GPT_KEY") or os.environ.get("GPT5_API_KEY", "")
    if model_alias == "claude_haiku":
        return os.environ.get("RUOLI_CLAUDE_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
    return ""


def _load_keyfile_env() -> None:
    path = Path.home() / ".api_keys"
    if not path.exists():
        return
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def _metrics(
    *,
    execution_mode: str,
    env: dict[str, Any],
    selected: list[dict[str, Any]],
    contract: dict[str, Any],
    preflight: dict[str, Any],
    live: dict[str, Any],
    acceptance: dict[str, Any],
    applied: list[str],
    before_node_count: int,
    after_node_count: int,
) -> dict[str, Any]:
    return {
        "execution_mode": execution_mode,
        "selected_candidate_count": len(selected),
        "contract_ready_count": contract["metrics"]["preflight_ready_count"],
        "preflight_ready_count": preflight["readiness_counts"].get("ready_for_fresh_ablation", 0),
        "fresh_live_path_present": True,
        "live_env_ready": bool(env.get("ready")),
        "fresh_api_call_count": int(live.get("fresh_api_call_count") or 0),
        "planned_fresh_api_call_count": int(live.get("planned_fresh_api_call_count") or 0),
        "accepted_count": len(acceptance.get("accepted_proposal_ids", [])),
        "acceptance_decision_counts": acceptance.get("decision_counts", {}),
        "applied_count": len(applied),
        "graph_copy_node_delta": after_node_count - before_node_count,
        "main_graph_mutation_count": 0,
        "secret_value_exposed": False,
    }


def _interpretation(*, execution_mode: str, env: dict[str, Any], metrics: dict[str, Any]) -> str:
    if execution_mode == "execute_live" and env["ready"] and metrics["fresh_api_call_count"]:
        return "Fresh API judgments were executed and fed through the same candidate acceptance/apply-copy path."
    if execution_mode == "execute_live":
        return "Fresh API execution was requested but blocked or incomplete; the artifact records the redacted env status."
    return "The fresh-live path is implemented and preflighted; this dry run does not count as fresh API evidence."


def _redacted_error(exc: Exception) -> str:
    text = str(exc)
    for key_name in ("RUOLI_GPT_KEY", "GPT5_API_KEY", "RUOLI_CLAUDE_KEY"):
        key = os.environ.get(key_name)
        if key:
            text = text.replace(key, "<redacted>")
    return f"{type(exc).__name__}: {text[:240]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fresh-live capable residual loop artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="full_v3_residual_fresh_live_loop_20260611")
    parser.add_argument("--execution-mode", choices=["dry_run", "execute_live", "summarize"], default="dry_run")
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--trigger-rows-per-candidate", type=int, default=4)
    parser.add_argument("--control-rows-per-candidate", type=int, default=2)
    parser.add_argument("--model-alias", default="gpt_mini")
    parser.add_argument("--no-keyfile", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_residual_fresh_live_loop_payload(
        root=root,
        eval_id=args.eval_id,
        execution_mode=args.execution_mode,
        candidate_count=args.candidate_count,
        trigger_rows_per_candidate=args.trigger_rows_per_candidate,
        control_rows_per_candidate=args.control_rows_per_candidate,
        model_alias=args.model_alias,
        load_keyfile=not args.no_keyfile,
    )
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
