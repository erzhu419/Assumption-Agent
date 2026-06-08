"""Live-ready positive queue for orthogonal hypothesis validation.

The structural orthogonal ablations prove that the gate can retain a new
explanation axis.  This module adds the missing positive arm before live model
spend: synthesize one orthogonal candidate, prove that novelty ON/OFF changes
only that axis, and prove that the candidate has enough trigger/control rows for
fresh ablation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .candidate_eval import CandidateReadiness, build_candidate_eval_payload
from .graph_memory import JsonlGraphStore
from .novelty_integration import NoveltyClass, build_novelty_integration_payload
from .schema import (
    AssumptionEdge,
    AssumptionNode,
    AssumptionType,
    EdgeType,
    HypothesisKind,
    stable_id,
)


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_GRAPH_DIR = Path("phase four/assumption_graph")
DEFAULT_SAMPLE_PATH = Path("phase two/analysis/cache/sample_100.json")
DEFAULT_META_PATH = Path("phase two/analysis/cache/answers/phase2_v20_meta.json")
DEFAULT_OUT = PAPER_DIR / "orthogonal_positive_queue_20260608.json"
DEFAULT_PROPOSALS_OUT = PAPER_DIR / "orthogonal_positive_queue_proposals_20260608.json"
DEFAULT_PREFLIGHT_OUT = PAPER_DIR / "orthogonal_positive_queue_preflight_20260608.json"

DEFAULT_TRIGGER_IDS = [
    "business_0097",
    "engineering_0244",
    "mathematics_0082",
    "engineering_0120",
]


def build_orthogonal_positive_queue_payload(
    *,
    root: Path,
    graph_dir: Path | None = None,
    sample_path: Path | None = None,
    meta_path: Path | None = None,
    eval_id: str | None = None,
    proposals_out: Path | None = None,
    preflight_out: Path | None = None,
) -> dict[str, Any]:
    """Build a positive, live-ready orthogonal candidate queue artifact."""

    root = root.resolve()
    graph_dir = _resolve(root, graph_dir or DEFAULT_GRAPH_DIR)
    sample_path = _resolve(root, sample_path or DEFAULT_SAMPLE_PATH)
    meta_path = _resolve(root, meta_path or DEFAULT_META_PATH)
    proposals_out = _resolve(root, proposals_out or DEFAULT_PROPOSALS_OUT)
    preflight_out = _resolve(root, preflight_out or DEFAULT_PREFLIGHT_OUT)
    eval_id = eval_id or "orthogonal_positive_queue_20260608"
    proposal_payload = _build_positive_proposal_payload(eval_id=eval_id)
    store = JsonlGraphStore(graph_dir)
    novelty_enabled = build_novelty_integration_payload(
        store,
        proposal_payload,
        eval_id=f"{eval_id}_novelty_enabled",
        enable_orthogonal=True,
    )
    novelty_disabled = build_novelty_integration_payload(
        store,
        proposal_payload,
        eval_id=f"{eval_id}_novelty_disabled",
        enable_orthogonal=False,
    )
    preflight = build_candidate_eval_payload(
        graph_dir=graph_dir,
        proposal_payload=proposal_payload,
        sample=_load_json(sample_path),
        meta_by_pid=_load_json(meta_path),
        eval_id=f"{eval_id}_preflight",
        proposal_ids=[proposal_payload["proposals"][0]["proposal_id"]],
        min_trigger_n=3,
        min_active_trigger_n=3,
        force_proposal_route=True,
        proposals_arg=_display_path(root, proposals_out),
        sample_arg=_display_path(root, sample_path),
    )
    summary = preflight["summaries"][0] if preflight.get("summaries") else {}
    enabled_row = novelty_enabled["rows"][0] if novelty_enabled.get("rows") else {}
    disabled_row = novelty_disabled["rows"][0] if novelty_disabled.get("rows") else {}
    env = _env_status()
    next_commands = _next_commands(
        root,
        summary,
        proposal_payload["proposals"][0]["proposal_id"],
        proposals_out,
        preflight_out,
        sample_path,
    )
    metrics = {
        "proposal_count": len(proposal_payload.get("proposals", [])),
        "enabled_orthogonal_count": novelty_enabled.get("classification_counts", {}).get(
            NoveltyClass.ORTHOGONAL_NEW_FAMILY.value,
            0,
        ),
        "disabled_orthogonal_count": novelty_disabled.get("classification_counts", {}).get(
            NoveltyClass.ORTHOGONAL_NEW_FAMILY.value,
            0,
        ),
        "enabled_orthogonal_edge_count": novelty_enabled.get("recommended_edge_counts", {}).get(
            EdgeType.ORTHOGONAL_TO.value,
            0,
        ),
        "disabled_orthogonal_edge_count": novelty_disabled.get("recommended_edge_counts", {}).get(
            EdgeType.ORTHOGONAL_TO.value,
            0,
        ),
        "preflight_ready_count": preflight.get("readiness_counts", {}).get(
            CandidateReadiness.READY_FOR_FRESH_ABLATION.value,
            0,
        ),
        "trigger_count": len(summary.get("trigger_problem_ids") or []),
        "active_trigger_count": len(summary.get("active_trigger_problem_ids") or []),
        "control_count": len(summary.get("control_problem_ids") or []),
        "outside_active_count": len(summary.get("outside_active_problem_ids") or []),
        "live_env_ready": env["gpt"]["ready"] or env["ruoli_gpt"]["ready"],
    }
    gates = {
        "proposal_artifact_is_single_positive_candidate": metrics["proposal_count"] == 1,
        "orthogonal_enabled_classifies_new_family": (
            enabled_row.get("classification") == NoveltyClass.ORTHOGONAL_NEW_FAMILY.value
        ),
        "orthogonal_disabled_collapses_to_specialization": (
            disabled_row.get("classification") != NoveltyClass.ORTHOGONAL_NEW_FAMILY.value
            and not bool(disabled_row.get("is_new_family"))
        ),
        "orthogonal_edge_only_when_enabled": (
            metrics["enabled_orthogonal_edge_count"] == 1
            and metrics["disabled_orthogonal_edge_count"] == 0
        ),
        "preflight_ready_for_fresh_ablation": metrics["preflight_ready_count"] == 1,
        "trigger_rows_present": metrics["trigger_count"] >= 3 and metrics["active_trigger_count"] >= 3,
        "control_rows_present": metrics["control_count"] >= 3,
        "no_no_fire_exposure": metrics["outside_active_count"] == 0,
        "commands_are_secret_free": _commands_are_secret_free(next_commands),
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "orthogonal_positive_live_ready_queue",
        "performance_validation": True,
        "validation_scope": (
            "positive orthogonal-new-family proposal classification plus trigger/control preflight; "
            "live answer-quality judging is queued and requires API environment variables"
        ),
        "status": "live_ready" if metrics["live_env_ready"] else "live_ready_env_missing",
        "pass": all(gates.values()),
        "source": {
            "root": ".",
            "graph_dir": _display_path(root, graph_dir),
            "sample_path": _display_path(root, sample_path),
            "meta_path": _display_path(root, meta_path),
            "proposals_out": _display_path(root, proposals_out),
            "preflight_out": _display_path(root, preflight_out),
        },
        "proposal_payload": proposal_payload,
        "novelty_enabled_summary": _condition_summary(novelty_enabled),
        "novelty_disabled_summary": _condition_summary(novelty_disabled),
        "novelty_rows": {
            "enabled": enabled_row,
            "disabled": disabled_row,
        },
        "preflight_summary": summary,
        "preflight_payload": preflight,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "env_status": env,
        "next_commands": next_commands,
        "interpretation": (
            "This is the positive queue complement to the judged negative control: a real graph parent and real "
            "sample/meta files produce an orthogonal_new_family candidate with an orthogonal_to edge and enough "
            "trigger/control rows for fresh ablation.  It is not yet a downstream win claim; that requires the "
            "queued live answer and judge commands to run with environment-provided API credentials."
        ),
    }


def _build_positive_proposal_payload(*, eval_id: str) -> dict[str, Any]:
    parent_id = "strategy_S01"
    candidate_id = stable_id("cand", eval_id, parent_id, "rubric_drift")
    proposal_id = stable_id("prop", eval_id, candidate_id)
    candidate = AssumptionNode(
        id=candidate_id,
        type=AssumptionType.EVALUATOR,
        kind=HypothesisKind.EVALUATOR_POLICY,
        claim=(
            "Before repairing the task strategy, test whether cross-judge rubric drift or stale acceptance "
            "feedback caused the residual; calibrate the evaluator axis against trigger and placebo controls."
        ),
        context_conditions=[
            "same failure residual as a strategy repair candidate",
            "answer content looks plausible but acceptance flips across judge/model/rubric variants",
        ],
        predicted_effects=[
            "avoid promoting strategy edits when the failure is caused by evaluator drift",
            "improve recursive retention by separating evaluator defects from method defects",
        ],
        risk_predictions=[
            "may waste ablation budget if the residual is actually a method defect",
            "may overfit to judge phrasing unless placebo controls pass",
        ],
        verifiers=[
            "cross_judge_disagreement_probe",
            "trigger_control_fresh_ablation",
            "placebo_rubric_stability_check",
        ],
        residual_ids=["res_orthogonal_positive_queue_evaluator_drift"],
        confidence=0.41,
        metaproductivity=0.06,
        status="candidate",
        tags=[
            "candidate",
            "orthogonal",
            "rubric_drift",
            "cross_judge_calibration",
            "acceptance_noise",
        ],
        source_refs=[
            f"parent:{parent_id}",
            "orthogonal_positive_queue",
            "reconstruction/md/orthogonal_hypothesis_gate_20260608.md",
        ],
        payload={
            "orthogonal_to_existing": True,
            "activation": {
                "problem_ids": DEFAULT_TRIGGER_IDS,
                "min_keyword_hits": 1,
            },
            "proposal_type": "orthogonal_failure_hypothesis",
            "residual_cluster": "evaluator_drift_vs_strategy_repair",
            "variation_evaluation_retention": {
                "variation": "new evaluator-axis hypothesis for the same strategy residual",
                "evaluation": "novelty ON/OFF gate plus trigger/control fresh-ablation preflight",
                "selective_retention": "retain only if downstream judged trigger benefit clears controls",
            },
        },
    )
    edge = AssumptionEdge(
        source=candidate_id,
        target=parent_id,
        type=EdgeType.GENERATED_FROM_RESIDUAL,
        weight=0.52,
        evidence="orthogonal_positive_queue",
        payload={
            "source": "orthogonal_positive_queue",
            "reason": "same residual/parent, different explanatory axis",
        },
    )
    proposal = {
        "proposal_id": proposal_id,
        "proposal_type": "orthogonal_failure_hypothesis",
        "parent_node_id": parent_id,
        "candidate_node": candidate.to_dict(),
        "edges": [edge.to_dict()],
        "manifest": None,
        "rationale": (
            "The current residual may not be a controlled-variable method defect.  It may be an evaluator/rubric "
            "defect, so test that orthogonal explanatory axis before mutating the strategy family."
        ),
        "priority": 0.72,
        "source_action": {
            "action_type": "orthogonal_positive_queue",
            "parent_node_id": parent_id,
            "trigger_problem_ids": DEFAULT_TRIGGER_IDS,
        },
    }
    return {
        "eval_id": eval_id,
        "source_eval_id": "orthogonal_positive_queue_builder",
        "proposal_counts": {"orthogonal_failure_hypothesis": 1},
        "proposals": [proposal],
    }


def _condition_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval_id": payload.get("eval_id"),
        "orthogonal_gate_enabled": payload.get("orthogonal_gate_enabled"),
        "classification_counts": payload.get("classification_counts"),
        "recommended_edge_counts": payload.get("recommended_edge_counts"),
        "pass": payload.get("pass"),
    }


def _env_status() -> dict[str, Any]:
    specs = {
        "gpt": ["GPT5_API_KEY", "GPT5_BASE_URL"],
        "ruoli_gpt": ["RUOLI_GPT_KEY", "RUOLI_BASE_URL"],
        "gemini": ["GEMINI_API_KEY", "GEMINI_BASE_URL"],
        "claude": ["ANTHROPIC_API_KEY"],
    }
    return {
        name: {
            "required_names": names,
            "set_names": [var for var in names if bool(os.environ.get(var))],
            "ready": all(bool(os.environ.get(var)) for var in names),
        }
        for name, names in specs.items()
    }


def _next_commands(
    root: Path,
    summary: dict[str, Any],
    proposal_id: str,
    proposals_out: Path,
    preflight_out: Path,
    sample_path: Path,
) -> list[dict[str, str]]:
    command_hint = str(summary.get("command_hint") or "")
    answer_command = (
        "RUOLI_GPT_KEY=<set-in-env> RUOLI_BASE_URL=<set-in-env> "
        "GPT_MINI_MODEL=gpt-5.4-mini "
        f"{command_hint}"
    )
    acceptance_command = (
        "python3 -m assumption_os.candidate_acceptance "
        f"--root . --proposals '{_display_path(root, proposals_out)}' "
        f"--preflight '{_display_path(root, preflight_out)}' "
        "--judgments '<judgments-json-from-pairwise-judge>' "
        f"--candidate-variant proposal_{proposal_id.replace('prop_', '')} "
        "--baseline-variant phase2_v20_gpt54mini_prop_union "
        f"--eval-id acceptance_{proposal_id}_orthogonal_positive_queue --proposal-ids {proposal_id} "
        "--summary-out '<acceptance-summary-json>'"
    )
    return [
        {
            "name": "fresh_ablation_answers",
            "command": answer_command,
        },
        {
            "name": "acceptance_gate_after_pairwise_judgments",
            "command": acceptance_command,
        },
    ]


def _commands_are_secret_free(commands: list[dict[str, str]]) -> bool:
    text = json.dumps(commands, ensure_ascii=False)
    return "sk-" not in text and "newapi_channel_conn" not in text and "<set-in-env>" in text


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a live-ready positive orthogonal proposal queue.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--sample", default=str(DEFAULT_SAMPLE_PATH))
    parser.add_argument("--meta", default=str(DEFAULT_META_PATH))
    parser.add_argument("--eval-id", default="orthogonal_positive_queue_20260608")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--proposals-out", default=str(DEFAULT_PROPOSALS_OUT))
    parser.add_argument("--preflight-out", default=str(DEFAULT_PREFLIGHT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    proposals_out = _resolve(root, Path(args.proposals_out))
    preflight_out = _resolve(root, Path(args.preflight_out))
    payload = build_orthogonal_positive_queue_payload(
        root=root,
        graph_dir=Path(args.graph_dir),
        sample_path=Path(args.sample),
        meta_path=Path(args.meta),
        eval_id=args.eval_id,
        proposals_out=proposals_out,
        preflight_out=preflight_out,
    )
    _write_json(proposals_out, payload["proposal_payload"])
    _write_json(preflight_out, payload["preflight_payload"])
    out = _resolve(root, Path(args.out))
    _write_json(out, payload)
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "status": payload["status"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
        "proposals_out": str(proposals_out),
        "preflight_out": str(preflight_out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
