"""Capability scoreboard for the Assumption OS loop.

The reconstruction plan explicitly warns against judging the system only by
answer win-rate.  This module turns the core lifecycle into separate capability
scores: explicit assumptions, context selection, execution fidelity, residual
attribution, memory transfer, metaproductivity, verifier reliability,
world-model quality, and harness governance.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .graph_memory import JsonlGraphStore


@dataclass(frozen=True)
class CapabilityScore:
    name: str
    score: float
    passed: bool
    threshold: float
    evidence: dict = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def build_assumption_bench_payload(
    *,
    eval_id: str,
    sections: dict[str, dict],
    graph_dir: Path | None = None,
    threshold: float = 0.72,
) -> dict:
    graph_stats = _graph_stats(graph_dir) if graph_dir else {}
    scores = [
        _score_assumption_explicitness(sections),
        _score_context_selection(sections),
        _score_execution_fidelity(sections),
        _score_residual_attribution(sections),
        _score_memory_transfer(sections, graph_stats),
        _score_metaproductivity(sections, graph_stats),
        _score_verifier_reliability(sections),
        _score_world_model_quality(sections),
        _score_harness_governance(sections),
    ]
    mean_score = round(sum(score.score for score in scores) / len(scores), 4) if scores else 0.0
    min_score = min((score.score for score in scores), default=0.0)
    return {
        "eval_id": eval_id,
        "threshold": threshold,
        "overall_score": mean_score,
        "min_score": round(min_score, 4),
        "pass": mean_score >= threshold and all(score.passed for score in scores),
        "capability_count": len(scores),
        "passed_capability_count": sum(1 for score in scores if score.passed),
        "failed_capabilities": [score.name for score in scores if not score.passed],
        "scores": [score.to_dict() for score in scores],
        "graph_stats": graph_stats,
    }


def _score_assumption_explicitness(sections: dict[str, dict]) -> CapabilityScore:
    manifest = sections.get("manifest_logger", {})
    runtime_trace = sections.get("runtime_trace", {})
    runtime_trace_events = int(runtime_trace.get("event_count") or 0)
    event_count = int(manifest.get("event_count") or 0) + runtime_trace_events
    real_log_count = int(manifest.get("real_log_event_count") or 0)
    no_leak = (
        not manifest.get("secret_leak_detected", True)
        and not runtime_trace.get("secret_leak_detected", False)
    )
    score = 0.5 * _cap(event_count / 100) + 0.3 * _cap(real_log_count / 10) + 0.2 * float(no_leak)
    return _capability(
        "assumption_explicitness",
        score,
        evidence={
            "event_count": event_count,
            "real_log_event_count": real_log_count,
            "runtime_trace_event_count": runtime_trace_events,
            "secret_leak_detected": not no_leak,
        },
        rationale="Key LLM/retrieval/judge/tool/simulator actions become redacted manifests.",
    )


def _score_context_selection(sections: dict[str, dict]) -> CapabilityScore:
    trajectory = sections.get("trajectory_search", {})
    verifier = sections.get("verifier_stack", {})
    multi_path = float(trajectory.get("multi_path_rate") or 0.0)
    hit = float(trajectory.get("top_path_label_hit_rate") or 0.0)
    proposal_count = int(verifier.get("proposal_count") or 0)
    score = 0.4 * _cap(multi_path / 0.8) + 0.4 * _cap(hit) + 0.2 * _cap(proposal_count / 16)
    return _capability(
        "context_selection",
        score,
        evidence={"multi_path_rate": multi_path, "top_path_label_hit_rate": hit, "proposal_count": proposal_count},
        rationale="The loop keeps multiple hypothesis futures alive and selects among enough candidates.",
    )


def _score_execution_fidelity(sections: dict[str, dict]) -> CapabilityScore:
    audit = sections.get("recursive_audit", {})
    daemon = sections.get("recursive_daemon", {})
    closure = float(audit.get("min_closure_score") or 0.0)
    critical = int(audit.get("critical_issue_count") or 0)
    case_count = int(daemon.get("case_count") or 0)
    applied = int(daemon.get("accepted_apply_count") or 0)
    apply_ratio = applied / case_count if case_count else 0.0
    score = 0.55 * closure + 0.25 * _cap(apply_ratio) + 0.2 * float(critical == 0 and bool(audit.get("pass")))
    return _capability(
        "execution_fidelity",
        score,
        evidence={"closure_score": closure, "critical_issue_count": critical, "accepted_apply_count": applied, "case_count": case_count},
        rationale="Recursive actions expose parent returns and daemon apply stays gated.",
    )


def _score_residual_attribution(sections: dict[str, dict]) -> CapabilityScore:
    residual = sections.get("residual_clusterer", {})
    clusters = int(residual.get("cluster_count") or 0)
    proposals = int(residual.get("proposal_count") or 0)
    plans = bool(residual.get("validation_plans_complete"))
    score = 0.4 * _cap(clusters / 5) + 0.3 * _cap(proposals / 2) + 0.3 * float(plans)
    return _capability(
        "residual_attribution",
        score,
        evidence={"cluster_count": clusters, "proposal_count": proposals, "validation_plans_complete": plans},
        rationale="Systematic residuals are clustered before new method hypotheses are synthesized.",
    )


def _score_memory_transfer(sections: dict[str, dict], graph_stats: dict) -> CapabilityScore:
    harness = sections.get("harness_observer", {})
    coverage = bool(harness.get("full_coverage_after_writeback"))
    artifacts = int(harness.get("artifact_file_count") or 0)
    node_types = int(graph_stats.get("node_type_count") or 0)
    edge_types = int(graph_stats.get("edge_type_count") or 0)
    score = (
        0.35 * float(coverage)
        + 0.2 * _cap(artifacts / 4)
        + 0.25 * _cap(node_types / 6)
        + 0.2 * _cap(edge_types / 6)
    )
    return _capability(
        "memory_transfer",
        score,
        evidence={"artifact_coverage": coverage, "artifact_file_count": artifacts, "node_type_count": node_types, "edge_type_count": edge_types},
        rationale="The graph has typed memory surfaces and observed artifacts that can transfer across runs.",
    )


def _score_metaproductivity(sections: dict[str, dict], graph_stats: dict) -> CapabilityScore:
    trajectory = sections.get("trajectory_search", {})
    selected = trajectory.get("selected_path_types", {})
    selected_diversity = len([k for k, v in selected.items() if v])
    positive_meta = int(graph_stats.get("positive_metaproductivity_nodes") or 0)
    score = 0.45 * _cap(selected_diversity / 3) + 0.35 * _cap(positive_meta / 20) + 0.2 * _cap(float(trajectory.get("multi_path_rate") or 0.0) / 0.8)
    return _capability(
        "metaproductivity",
        score,
        evidence={"selected_path_type_count": selected_diversity, "positive_metaproductivity_nodes": positive_meta, "multi_path_rate": trajectory.get("multi_path_rate")},
        rationale="The selector preserves productive descendants instead of only immediate wins.",
    )


def _score_verifier_reliability(sections: dict[str, dict]) -> CapabilityScore:
    verifier = sections.get("verifier_stack", {})
    stage_counts = verifier.get("stage_status_counts", {})
    v4_pass = int(stage_counts.get("V4:pass") or 0)
    v4_fail = int(stage_counts.get("V4:fail") or 0)
    experiments = int(verifier.get("falsification_experiment_count") or 0)
    score = (
        0.25 * float(bool(verifier.get("accepted_protocol_ok")))
        + 0.25 * float(bool(verifier.get("rejected_protocol_ok")))
        + 0.25 * _cap(experiments / 100)
        + 0.25 * _cap((v4_pass + v4_fail) / 16)
    )
    return _capability(
        "verifier_reliability",
        score,
        evidence={"accepted_protocol_ok": verifier.get("accepted_protocol_ok"), "rejected_protocol_ok": verifier.get("rejected_protocol_ok"), "falsification_experiment_count": experiments, "v4_pass": v4_pass, "v4_fail": v4_fail},
        rationale="Verifier decisions include positive, negative, and falsification-protocol evidence.",
    )


def _score_world_model_quality(sections: dict[str, dict]) -> CapabilityScore:
    world = sections.get("world_model", {})
    trace_dataset = sections.get("trace_dataset", {})
    trace_outcome = sections.get("trace_outcome_model", {})
    pre = world.get("pre_acceptance", {})
    calibration = world.get("post_calibration", {})
    auc = float(pre.get("auc") or 0.0)
    brier = float(calibration.get("brier_score") if calibration.get("brier_score") is not None else 1.0)
    labels = int(world.get("matched_label_count") or 0)
    trace_rows = int(trace_dataset.get("trainable_row_count") or 0)
    trace_metrics = trace_outcome.get("leave_one_out_metrics", {})
    trace_brier = trace_metrics.get("brier_score")
    trace_quality = 0.0 if trace_brier is None else _cap(1.0 - min(float(trace_brier) / 0.25, 1.0))
    proposal_score = 0.45 * _cap(auc) + 0.35 * _cap(1.0 - min(brier / 0.1, 1.0)) + 0.2 * _cap((labels + trace_rows) / 16)
    trace_enhanced_score = (
        0.4 * _cap(auc)
        + 0.3 * _cap(1.0 - min(brier / 0.1, 1.0))
        + 0.15 * _cap((labels + trace_rows) / 16)
        + 0.15 * trace_quality
    )
    score = max(proposal_score, trace_enhanced_score)
    return _capability(
        "world_model_quality",
        score,
        evidence={
            "auc": auc,
            "brier_score": brier,
            "matched_label_count": labels,
            "trace_trainable_row_count": trace_rows,
            "trace_outcome_brier_score": trace_brier,
        },
        rationale="The cheap simulator ranks accepted candidates above rejected ones and calibrates after evidence.",
    )


def _score_harness_governance(sections: dict[str, dict]) -> CapabilityScore:
    context = sections.get("evolution_context", {})
    responsibilities = context.get("responsibility_status_counts", {})
    pass_count = int(responsibilities.get("pass") or 0)
    total = int(context.get("responsibility_count") or 9)
    blocked = context.get("blocked_policy_decision") == "blocked_by_permissions"
    gated = context.get("apply_policy_decision") == "gated_apply_allowed"
    score = 0.5 * _cap(pass_count / total if total else 0.0) + 0.25 * float(blocked) + 0.25 * float(gated)
    return _capability(
        "harness_governance",
        score,
        evidence={"responsibility_status_counts": responsibilities, "blocked_policy_decision": context.get("blocked_policy_decision"), "apply_policy_decision": context.get("apply_policy_decision")},
        rationale="The evolution procedure has explicit responsibilities and permission gates.",
    )


def _capability(name: str, score: float, *, evidence: dict, rationale: str, threshold: float = 0.72) -> CapabilityScore:
    score = round(_cap(score), 4)
    return CapabilityScore(
        name=name,
        score=score,
        passed=score >= threshold,
        threshold=threshold,
        evidence=evidence,
        rationale=rationale,
    )


def _graph_stats(graph_dir: Path | None) -> dict:
    if not graph_dir:
        return {}
    store = JsonlGraphStore(graph_dir)
    node_type_counts = Counter(str(getattr(node.type, "value", node.type)) for node in store.nodes.values())
    edge_type_counts = Counter(str(getattr(edge.type, "value", edge.type)) for edge in store.edges)
    return {
        "node_count": len(store.nodes),
        "edge_count": len(store.edges),
        "trial_count": len(store.trials),
        "node_type_count": len(node_type_counts),
        "edge_type_count": len(edge_type_counts),
        "node_type_counts": dict(node_type_counts),
        "edge_type_counts": dict(edge_type_counts),
        "positive_metaproductivity_nodes": sum(1 for node in store.nodes.values() if float(node.metaproductivity or 0.0) > 0.0),
    }


def _cap(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--performance-payload", required=True)
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--threshold", type=float, default=0.72)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    performance = _load_json(_resolve(root, args.performance_payload))
    payload = build_assumption_bench_payload(
        eval_id=args.eval_id,
        sections=performance.get("sections", performance),
        graph_dir=_resolve(root, args.graph_dir),
        threshold=args.threshold,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
