"""Multi-path hypothesis trajectory search.

The recursive runner gives the current frontier.  The world model estimates
candidate promise.  This module combines both into ranked possible futures so
the agent can choose among promote, repair, evidence, and reject paths instead
of following one hard-coded action.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .schema import stable_id


@dataclass(frozen=True)
class HypothesisTrajectory:
    trajectory_id: str
    proposal_id: str
    frame_id: str
    path_type: str
    actions: list[str]
    predicted_success_probability: float
    expected_utility: float
    cost_proxy: float
    risk_proxy: float
    score: float
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def build_trajectory_search_payload(
    *,
    recursive_payload: dict,
    world_model_payload: dict | None = None,
    eval_id: str,
    beam_width: int = 5,
    max_paths_per_candidate: int = 3,
) -> dict:
    """Return ranked multi-path futures for the recursive frontier."""

    predictions = {
        row.get("proposal_id"): row
        for row in (world_model_payload or {}).get("predictions", [])
        if row.get("proposal_id")
    }
    trajectories = []
    for action in recursive_payload.get("next_actions", []):
        proposal_id = _proposal_id(action)
        prediction = predictions.get(proposal_id, {})
        for path in _candidate_paths(action, prediction)[:max_paths_per_candidate]:
            trajectories.append(path)
    trajectories = sorted(trajectories, key=lambda t: (-t.score, t.proposal_id, t.path_type))
    selected = trajectories[:beam_width]
    return {
        "eval_id": eval_id,
        "source_recursive_eval_id": recursive_payload.get("eval_id"),
        "source_world_model_eval_id": (world_model_payload or {}).get("eval_id"),
        "mode": {
            "beam_width": beam_width,
            "max_paths_per_candidate": max_paths_per_candidate,
        },
        "frontier_actions": len(recursive_payload.get("next_actions", [])),
        "trajectory_count": len(trajectories),
        "selected_count": len(selected),
        "path_type_counts": dict(Counter(t.path_type for t in trajectories)),
        "selected_trajectory_ids": [t.trajectory_id for t in selected],
        "trajectories": [t.to_dict() for t in trajectories],
        "selected": [t.to_dict() for t in selected],
    }


def _candidate_paths(action: dict, prediction: dict) -> list[HypothesisTrajectory]:
    proposal_id = _proposal_id(action)
    frame_id = action.get("frame_id", "")
    current_action = action.get("next_action", "")
    priority = float(action.get("priority", 0.0) or 0.0)
    prob = float(prediction.get("predicted_acceptance_probability", 0.5) or 0.5)
    utility = float(prediction.get("expected_utility", 0.0) or 0.0)
    risk = _risk_proxy(prediction.get("predicted_regression_risk", "unknown"))
    recommended = prediction.get("recommended_next_action") or current_action
    failure_modes = prediction.get("predicted_failure_modes", [])

    paths = []
    base_rationale = [
        f"frontier_action={current_action}",
        f"world_model_action={recommended}",
        f"predicted_acceptance={prob:.3f}",
        f"risk={prediction.get('predicted_regression_risk', 'unknown')}",
    ]
    if failure_modes:
        base_rationale.append("failure_modes=" + ",".join(failure_modes[:4]))

    if recommended in {"run_fresh_ablation", "apply_accepted_candidate_if_requested"} or prob >= 0.68:
        paths.append(_trajectory(
            proposal_id=proposal_id,
            frame_id=frame_id,
            path_type="promote_after_verification",
            actions=_dedupe_actions([
                current_action,
                "world_model_screen",
                "run_fresh_ablation",
                "candidate_acceptance_gate",
                "gated_apply_if_accepted",
                "post_apply_regression_monitor",
            ]),
            prob=prob,
            utility=utility,
            priority=priority,
            cost=0.8,
            risk=risk,
            rationale=[*base_rationale, "best when probability is high and regression risk is controlled"],
        ))

    if recommended.startswith("repair") or risk >= 0.35 or prob < 0.68:
        repair_action = _repair_action(recommended, current_action, risk)
        paths.append(_trajectory(
            proposal_id=proposal_id,
            frame_id=frame_id,
            path_type="repair_then_retest",
            actions=_dedupe_actions([
                current_action,
                "world_model_screen",
                repair_action,
                "rerun_preflight",
                "run_fresh_ablation_if_ready",
            ]),
            prob=min(0.85, prob + 0.08),
            utility=utility + 0.08,
            priority=priority,
            cost=0.55,
            risk=max(0.05, risk - 0.12),
            rationale=[*base_rationale, "repair path reduces predicted failure mode before spending on ablation"],
        ))

    if current_action in {"collect_more_evidence", "collect_more_judgments"} or prob < 0.68 or risk >= 0.3:
        paths.append(_trajectory(
            proposal_id=proposal_id,
            frame_id=frame_id,
            path_type="evidence_first",
            actions=_dedupe_actions([
                current_action,
                "collect_more_trigger_rows",
                "rerun_conditioned_preflight",
                "rerun_world_model",
            ]),
            prob=max(0.5, prob),
            utility=utility * 0.7,
            priority=priority,
            cost=0.35,
            risk=max(0.02, risk - 0.05),
            rationale=[*base_rationale, "evidence-first path is cheaper when the current signal is underpowered"],
        ))

    if prob <= 0.42 or recommended == "reject_or_revise_candidate":
        paths.append(_trajectory(
            proposal_id=proposal_id,
            frame_id=frame_id,
            path_type="reject_and_synthesize",
            actions=_dedupe_actions([
                current_action,
                "record_rejection_manifest",
                "cluster_residuals",
                "synthesize_novel_method_hypothesis",
            ]),
            prob=1.0 - prob,
            utility=max(0.0, -utility) + 0.05,
            priority=priority,
            cost=0.25,
            risk=0.05,
            rationale=[*base_rationale, "candidate looks weak; preserve the residual as synthesis material"],
        ))

    if not paths:
        paths.append(_trajectory(
            proposal_id=proposal_id,
            frame_id=frame_id,
            path_type="defer",
            actions=[current_action or "defer_until_more_context"],
            prob=prob,
            utility=utility,
            priority=priority,
            cost=0.1,
            risk=risk,
            rationale=base_rationale,
        ))
    return sorted(paths, key=lambda t: -t.score)


def _trajectory(
    *,
    proposal_id: str,
    frame_id: str,
    path_type: str,
    actions: list[str],
    prob: float,
    utility: float,
    priority: float,
    cost: float,
    risk: float,
    rationale: list[str],
) -> HypothesisTrajectory:
    score = (0.5 * prob) + (0.35 * utility) + (0.25 * priority) - (0.2 * cost) - (0.35 * risk)
    return HypothesisTrajectory(
        trajectory_id=stable_id("traj", proposal_id, frame_id, path_type, ",".join(actions)),
        proposal_id=proposal_id,
        frame_id=frame_id,
        path_type=path_type,
        actions=actions,
        predicted_success_probability=round(max(0.0, min(1.0, prob)), 4),
        expected_utility=round(utility, 4),
        cost_proxy=round(cost, 4),
        risk_proxy=round(risk, 4),
        score=round(score, 4),
        rationale=rationale,
    )


def _proposal_id(action: dict) -> str:
    if action.get("proposal_id"):
        return action["proposal_id"]
    problem_id = action.get("problem_id", "")
    for prefix in ("verify::", "proposal::", "evidence::", "repair::"):
        if problem_id.startswith(prefix):
            return problem_id.removeprefix(prefix)
    return ""


def _risk_proxy(risk: str) -> float:
    return {"high": 0.6, "medium": 0.3, "low": 0.05}.get(risk, 0.2)


def _repair_action(recommended: str, current_action: str, risk: float) -> str:
    if "retrieval" in recommended or "retrieval" in current_action:
        return "repair_retrieval_policy"
    if "formal" in recommended:
        return "repair_formal_mapping"
    if risk >= 0.35 or "scope" in recommended or "scope" in current_action:
        return "narrow_activation_scope"
    return "revise_candidate_claim"


def _dedupe_actions(actions: list[str]) -> list[str]:
    out = []
    for action in actions:
        if not action or (out and out[-1] == action):
            continue
        out.append(action)
    return out


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--recursive-payload", required=True)
    ap.add_argument("--world-model-payload", default=None)
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--beam-width", type=int, default=5)
    ap.add_argument("--max-paths-per-candidate", type=int, default=3)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    payload = build_trajectory_search_payload(
        recursive_payload=_load_json(_resolve(root, args.recursive_payload)),
        world_model_payload=(
            _load_json(_resolve(root, args.world_model_payload))
            if args.world_model_payload
            else None
        ),
        eval_id=args.eval_id,
        beam_width=args.beam_width,
        max_paths_per_candidate=args.max_paths_per_candidate,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
