"""Progress audit against ``reconstruction.md``.

The reconstruction plan is intentionally broader than a single benchmark.  This
module turns it into an auditable capability matrix with two separate scores:

* structure: the mechanism exists as code/artifacts.
* behavior: the mechanism has validated runtime evidence.

The numbers are not a substitute for heldout performance.  They make the gap to
the reconstruction target explicit so the next repair step is selected from
evidence rather than from a loose narrative.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProgressItem:
    key: str
    target: str
    structure_score: float
    behavior_score: float
    evidence: dict[str, Any]
    remaining_gaps: list[str]
    next_actions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "target": self.target,
            "structure_score": round(self.structure_score, 4),
            "behavior_score": round(self.behavior_score, 4),
            "evidence": self.evidence,
            "remaining_gaps": self.remaining_gaps,
            "next_actions": self.next_actions,
            "status": _status(self.structure_score, self.behavior_score),
        }


def build_reconstruction_progress_payload(
    *,
    root: Path,
    performance_payload: dict[str, Any],
    graph_dir: Path,
    eval_id: str,
    reconstruction_path: Path | None = None,
) -> dict[str, Any]:
    sections = performance_payload.get("sections", {})
    graph_stats = _graph_stats(graph_dir)
    raw_items = [
        _graph_memory_item(sections, graph_stats),
        _hypothesis_generator_item(sections),
        _world_model_item(sections),
        _verifier_stack_item(sections),
        _residual_analyzer_item(sections),
        _metaproductivity_selector_item(sections, graph_stats),
        _formal_alignment_item(sections),
        _recursive_loop_item(sections),
        _evaluation_system_item(sections),
    ]
    items = [_apply_reconstruction_ceiling(item) for item in raw_items]
    structure = _mean(item.structure_score for item in items)
    behavior = _mean(item.behavior_score for item in items)
    weighted = round(0.45 * structure + 0.55 * behavior, 4)
    closure = {
        "structure_percent": round(structure * 100, 1),
        "behavior_percent": round(behavior * 100, 1),
        "weighted_percent": round(weighted * 100, 1),
        "completed_item_count": sum(1 for item in items if item.structure_score >= 0.85 and item.behavior_score >= 0.75),
        "item_count": len(items),
    }
    payload = {
        "eval_id": eval_id,
        "source": {
            "root": ".",
            "performance_eval_id": performance_payload.get("eval_id"),
            "graph_dir": _display_path(root, graph_dir),
            "reconstruction_path": _display_path(root, reconstruction_path) if reconstruction_path else None,
        },
        "reconstruction_reference": _reconstruction_reference(reconstruction_path),
        "overall_pass": behavior >= 0.65 and structure >= 0.75,
        "closure": closure,
        "items": [item.to_dict() for item in items],
        "remaining_gaps_ranked": _rank_remaining_gaps(items),
        "next_actions_ranked": _rank_next_actions(items),
    }
    return payload


def _graph_memory_item(sections: dict[str, dict], graph_stats: dict) -> ProgressItem:
    memory = sections.get("memory_surfaces", {})
    harness = sections.get("harness_observer", {})
    structure = _avg([
        _cap(graph_stats.get("node_type_count", 0) / 11),
        _cap(graph_stats.get("edge_type_count", 0) / 11),
        float(memory.get("pass", False)),
        float(harness.get("full_coverage_after_writeback", False)),
    ])
    behavior = _avg([
        float(memory.get("pass", False)),
        float(harness.get("pass", False)),
        _cap(graph_stats.get("trial_count", 0) / 100),
        _cap(memory.get("surface_count", 0) / 10),
    ])
    return ProgressItem(
        key="A_assumption_graph_memory",
        target="HippoRAG-style Assumption Graph with assumptions, cases, residuals, verifiers, and runtime surfaces.",
        structure_score=structure,
        behavior_score=behavior,
        evidence={
            "node_type_count": graph_stats.get("node_type_count"),
            "edge_type_count": graph_stats.get("edge_type_count"),
            "trial_count": graph_stats.get("trial_count"),
            "surface_count": memory.get("surface_count"),
            "artifact_coverage": harness.get("full_coverage_after_writeback"),
        },
        remaining_gaps=[
            "Graph retrieval is still local JSONL/simple scoring, not a full HippoRAG phrase/passage/PPR memory.",
            "Runtime trace coverage is broad but not yet exhaustive across every runner/tool call.",
        ],
        next_actions=[
            "Add graph retrieval regression tests over multi-hop residual/case/verifier paths.",
            "Expand first-party trace hooks to judge and daemon command execution paths.",
        ],
    )


def _hypothesis_generator_item(sections: dict[str, dict]) -> ProgressItem:
    residual = sections.get("residual_clusterer", {})
    trace_policy = sections.get("trace_policy_proposals", {})
    preflight = sections.get("trace_policy_preflight", {})
    structure = _avg([
        float(residual.get("pass", False)),
        float(trace_policy.get("pass", False)),
        _cap(residual.get("proposal_count", 0) / 2),
        _cap(trace_policy.get("proposal_count", 0) / 3),
    ])
    behavior = _avg([
        _cap(residual.get("cluster_count", 0) / 7),
        _cap(trace_policy.get("repair_policy_count", 0) / 1),
        _cap(preflight.get("ready_count", 0) / max(1, preflight.get("proposal_count", 1))),
        float(preflight.get("pass", False)),
    ])
    return ProgressItem(
        key="B_hypothesis_generator",
        target="Generate framing, strategy, retrieval, evaluator, world-model, and self-modification hypotheses from residuals and traces.",
        structure_score=structure,
        behavior_score=behavior,
        evidence={
            "residual_cluster_count": residual.get("cluster_count"),
            "residual_proposal_count": residual.get("proposal_count"),
            "trace_policy_proposal_count": trace_policy.get("proposal_count"),
            "trace_policy_ready_count": preflight.get("ready_count"),
        },
        remaining_gaps=[
            "Generation is still mostly deterministic/residual-driven; broad LLM synthesis is injectable but not routinely validated.",
            "Evaluator/world-model/self-modification hypothesis generators exist as surfaces but need more real proposal examples.",
        ],
        next_actions=[
            "Run fresh ablation/judge for the preflight-ready trace policy proposals.",
            "Add a generator pass that turns evaluator/world-model residuals into candidate proposals.",
        ],
    )


def _world_model_item(sections: dict[str, dict]) -> ProgressItem:
    world = sections.get("world_model", {})
    trace_dataset = sections.get("trace_dataset", {})
    trace_outcome = sections.get("trace_outcome_model", {})
    brier = world.get("post_calibration", {}).get("brier_score")
    trace_brier = trace_outcome.get("leave_one_out_metrics", {}).get("brier_score")
    weighted_trace_brier = trace_outcome.get("leave_one_out_metrics", {}).get("weighted_brier_score")
    weighted_trace_rows = max(
        float(trace_dataset.get("weighted_trainable_row_count", 0.0) or 0.0),
        float(trace_outcome.get("weighted_trainable_row_count", 0.0) or 0.0),
    )
    structure = _avg([
        float(world.get("pass", False)),
        float(trace_dataset.get("pass", False)),
        float(trace_outcome.get("pass", False)),
        _cap(world.get("matched_label_count", 0) / 16),
    ])
    behavior = _avg([
        _score_brier(brier, threshold=0.1),
        _score_brier(weighted_trace_brier if weighted_trace_brier is not None else trace_brier, threshold=0.25),
        _cap(weighted_trace_rows / 50),
        _cap(world.get("matched_label_count", 0) / 50),
    ])
    return ProgressItem(
        key="C_world_model_simulator",
        target="Cheap predictor for assumption utility, failure mode, residual type, and execution worthiness.",
        structure_score=structure,
        behavior_score=behavior,
        evidence={
            "proposal_label_count": world.get("matched_label_count"),
            "proposal_brier": brier,
            "trace_trainable_rows": trace_outcome.get("trainable_row_count"),
            "weighted_trace_trainable_rows": weighted_trace_rows,
            "artifact_replay_trainable_rows": trace_dataset.get("artifact_replay_trainable_row_count"),
            "trace_brier": trace_brier,
            "weighted_trace_brier": weighted_trace_brier,
            "trace_source_counts": trace_outcome.get("trace_source_counts"),
        },
        remaining_gaps=[
            "The predictor is calibrated on tens of labels, not the 1000+ distilled trajectories described in reconstruction.md.",
            "It predicts proposal/route outcomes, but not yet full draft/audit/final trajectory quality.",
        ],
        next_actions=[
            "Accumulate a larger trace dataset from real first-party runs.",
            "Train/calibrate a cheap predictor over problem + activated assumptions + trace features + residual label.",
        ],
    )


def _verifier_stack_item(sections: dict[str, dict]) -> ProgressItem:
    verifier = sections.get("verifier_stack", {})
    preflight = sections.get("trace_policy_preflight", {})
    structure = _avg([
        float(verifier.get("pass", False)),
        float(verifier.get("accepted_protocol_ok", False)),
        float(verifier.get("rejected_protocol_ok", False)),
        _cap(verifier.get("falsification_protocol_candidate_count", 0) / max(1, verifier.get("proposal_count", 1))),
    ])
    behavior = _avg([
        _cap(verifier.get("falsification_experiment_count", 0) / 135),
        _cap((verifier.get("accepted_count", 0) + verifier.get("rejected_count", 0)) / max(1, verifier.get("proposal_count", 1))),
        float(preflight.get("pass", False)),
        _cap(preflight.get("ready_count", 0) / max(1, preflight.get("proposal_count", 1))),
    ])
    return ProgressItem(
        key="D_verifier_stack",
        target="Layered V0-V6 verification with falsification, controls, fresh splits, and high-stakes gates.",
        structure_score=structure,
        behavior_score=behavior,
        evidence={
            "proposal_count": verifier.get("proposal_count"),
            "accepted_count": verifier.get("accepted_count"),
            "rejected_count": verifier.get("rejected_count"),
            "falsification_experiment_count": verifier.get("falsification_experiment_count"),
            "trace_preflight_ready_count": preflight.get("ready_count"),
        },
        remaining_gaps=[
            "V0-V4 are represented; objective benchmark V5 and human-review V6 are still policy/documentation rather than active gates.",
            "Trace policy proposals are preflight-ready but have not yet run fresh ablation/judge.",
        ],
        next_actions=[
            "Execute the trace policy proposal ablation queue with cached/low-cost samples first.",
            "Add objective-task and manual-review gates for high-impact graph mutations.",
        ],
    )


def _residual_analyzer_item(sections: dict[str, dict]) -> ProgressItem:
    residual = sections.get("residual_clusterer", {})
    trace_dataset = sections.get("trace_dataset", {})
    trace_outcome = sections.get("trace_outcome_model", {})
    structure = _avg([
        float(residual.get("pass", False)),
        float(trace_dataset.get("pass", False)),
        _cap(len(residual.get("residual_type_counts", {})) / 3),
        _cap(trace_outcome.get("residual_group_count", 0) / 1),
    ])
    behavior = _avg([
        _cap(residual.get("record_count", 0) / 100),
        _cap(residual.get("cluster_count", 0) / 7),
        _cap(residual.get("proposal_count", 0) / 2),
        float(residual.get("validation_plans_complete", False)),
    ])
    return ProgressItem(
        key="E_residual_analyzer",
        target="Classify failures into execution, optimization, assumption, memory, evaluator, and simulator residuals before mutation.",
        structure_score=structure,
        behavior_score=behavior,
        evidence={
            "residual_record_count": residual.get("record_count"),
            "cluster_count": residual.get("cluster_count"),
            "residual_type_counts": residual.get("residual_type_counts"),
            "validation_plans_complete": residual.get("validation_plans_complete"),
        },
        remaining_gaps=[
            "Residual labels are deterministic and artifact-derived; no calibrated residual classifier has been validated on human labels.",
            "Skipped/non-attributed failures are improved but not exhaustively covered by first-party traces.",
        ],
        next_actions=[
            "Add residual-label agreement tests using LLM/human-labeled examples.",
            "Ensure skipped and bypass losses always become trace residual rows.",
        ],
    )


def _metaproductivity_selector_item(sections: dict[str, dict], graph_stats: dict) -> ProgressItem:
    trajectory = sections.get("trajectory_search", {})
    bench = sections.get("assumption_bench", {})
    structure = _avg([
        float(trajectory.get("pass", False)),
        _cap(trajectory.get("multi_path_rate", 0.0) / 0.8),
        _cap(len(trajectory.get("selected_path_types", {})) / 4),
        _cap(graph_stats.get("positive_metaproductivity_nodes", 0) / 100),
    ])
    behavior = _avg([
        _cap(trajectory.get("top_path_label_hit_rate", 0.0)),
        _cap(trajectory.get("trajectory_count", 0) / 26),
        _cap(trajectory.get("frontier_actions", 0) / 10),
        _cap(bench.get("score_by_capability", {}).get("metaproductivity", 0.0)),
    ])
    return ProgressItem(
        key="F_metaproductivity_selector",
        target="Select assumption families by utility plus long-run clade productivity, cost, risk, and novelty.",
        structure_score=structure,
        behavior_score=behavior,
        evidence={
            "multi_path_rate": trajectory.get("multi_path_rate"),
            "trajectory_count": trajectory.get("trajectory_count"),
            "top_path_label_hit_rate": trajectory.get("top_path_label_hit_rate"),
            "positive_metaproductivity_nodes": graph_stats.get("positive_metaproductivity_nodes"),
        },
        remaining_gaps=[
            "Metaproductivity is scored in graph/trajectory artifacts, but ACP is not yet a learned long-horizon value model.",
            "Cost/risk/novelty are present as features but not optimized as a unified scheduler objective.",
        ],
        next_actions=[
            "Use accepted/rejected descendants to update clade-level ACP after each recursive run.",
            "Add a scheduler benchmark that compares ACP-aware selection to immediate-win selection.",
        ],
    )


def _formal_alignment_item(sections: dict[str, dict]) -> ProgressItem:
    formal = sections.get("formal_metrics", {})
    structure = _avg([
        float(formal.get("pass", False)),
        _cap(formal.get("mapping_count", 0) / 9),
        float(formal.get("dedup_pass", False)),
        float(formal.get("transfer_eval_pass", False)),
        float(formal.get("warning_count", 1) == 0),
    ])
    behavior = _avg([
        _cap(formal.get("complete_count", 0) / 9),
        _cap(formal.get("same_shape_count", 0) / max(1, formal.get("mapping_count", 1))),
        float(formal.get("warning_count", 1) == 0),
        float(formal.get("dedup_pass", False)),
        _cap(formal.get("dedup_complete_mapping_count", 0) / max(1, formal.get("mapping_count", 1))),
        _cap(formal.get("transfer_pairwise_auc", 0.0)),
        _cap(formal.get("transfer_top1_hit_rate", 0.0)),
    ])
    return ProgressItem(
        key="G_formal_alignment_layer",
        target="Use bounded category/info-geometry mappings for deduplication, cross-domain transfer, and promotion-sensitive policy checks.",
        structure_score=structure,
        behavior_score=behavior,
        evidence={
            "mapping_count": formal.get("mapping_count"),
            "complete_count": formal.get("complete_count"),
            "same_shape_count": formal.get("same_shape_count"),
            "warning_count": formal.get("warning_count"),
            "dedup_unique_signature_count": formal.get("dedup_unique_signature_count"),
            "dedup_duplicate_cluster_count": formal.get("dedup_duplicate_cluster_count"),
            "dedup_positive_control": formal.get("dedup_positive_control"),
            "transfer_top1_hit_rate": formal.get("transfer_top1_hit_rate"),
            "transfer_pairwise_auc": formal.get("transfer_pairwise_auc"),
        },
        remaining_gaps=[
            "Formal mapping is an audit/gate over finite kernels, not a full category-theoretic or information-geometric reasoning engine.",
            "Formal transfer correlation is validated on a small labeled query audit, not a broad downstream task suite.",
        ],
        next_actions=[
            "Expand formal-transfer labels beyond the current five-query audit.",
            "Use dedup recommendations to merge complete formal equivalents after verifier approval.",
        ],
    )


def _recursive_loop_item(sections: dict[str, dict]) -> ProgressItem:
    audit = sections.get("recursive_audit", {})
    daemon = sections.get("recursive_daemon", {})
    trace_preflight = sections.get("trace_policy_preflight", {})
    structure = _avg([
        float(audit.get("pass", False)),
        float(daemon.get("pass", False)),
        _cap(audit.get("min_closure_score", 0.0)),
        _cap(trace_preflight.get("ready_count", 0) / max(1, trace_preflight.get("proposal_count", 1))),
    ])
    behavior = _avg([
        _cap(audit.get("actionable_count", 0) / 5),
        _cap(daemon.get("accepted_apply_count", 0) / max(1, daemon.get("case_count", 1))),
        float(audit.get("critical_issue_count", 1) == 0),
        float(trace_preflight.get("pass", False)),
    ])
    return ProgressItem(
        key="recursive_execution_loop",
        target="Problem -> retrieve -> generate trajectories -> simulate -> select -> act -> manifest -> residual -> update/generate.",
        structure_score=structure,
        behavior_score=behavior,
        evidence={
            "recursive_closure_score": audit.get("min_closure_score"),
            "actionable_count": audit.get("actionable_count"),
            "daemon_cases": daemon.get("case_count"),
            "accepted_apply_count": daemon.get("accepted_apply_count"),
            "trace_policy_ready_count": trace_preflight.get("ready_count"),
        },
        remaining_gaps=[
            "The loop is executable and gated, but not yet an unattended daemon running continuous fresh ablation/judge cycles.",
            "Actual graph mutation remains correctly gated by explicit apply/writeback permissions.",
        ],
        next_actions=[
            "Let the recursive daemon consume the preflight-ready trace policy proposals and record planned leaf commands.",
            "Run one bounded fresh-ablation cycle and resume parents from judgments.",
        ],
    )


def _evaluation_system_item(sections: dict[str, dict]) -> ProgressItem:
    bench = sections.get("assumption_bench", {})
    structure = _avg([
        float(bench.get("pass", False)),
        _cap(bench.get("passed_capability_count", 0) / max(1, bench.get("capability_count", 1))),
        _cap(len(bench.get("score_by_capability", {})) / 9),
    ])
    behavior = _avg([
        _cap(bench.get("overall_score", 0.0)),
        _cap(bench.get("min_score", 0.0)),
        float(not bench.get("failed_capabilities")),
    ])
    return ProgressItem(
        key="assumption_bench_evaluation",
        target="Evaluate lifecycle capabilities separately from pooled answer win-rate.",
        structure_score=structure,
        behavior_score=behavior,
        evidence={
            "overall_score": bench.get("overall_score"),
            "min_score": bench.get("min_score"),
            "capability_count": bench.get("capability_count"),
            "passed_capability_count": bench.get("passed_capability_count"),
        },
        remaining_gaps=[
            "Capability scores are strong, but larger heldout answer-quality validation is still needed.",
            "Residual classification and formal-transfer scores need independent labels.",
        ],
        next_actions=[
            "Run a larger heldout performance slice after each accepted mechanism update.",
            "Add labeled diagnostic datasets for residual classification and formal transfer.",
        ],
    )


def _rank_remaining_gaps(items: list[ProgressItem]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        gap_weight = 1.0 - item.behavior_score
        for gap in item.remaining_gaps:
            rows.append({
                "item": item.key,
                "gap": gap,
                "priority": round(gap_weight, 4),
            })
    return sorted(rows, key=lambda row: (-row["priority"], row["item"], row["gap"]))[:12]


def _rank_next_actions(items: list[ProgressItem]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        priority = 1.0 - 0.55 * item.behavior_score - 0.45 * item.structure_score
        for action in item.next_actions:
            rows.append({
                "item": item.key,
                "action": action,
                "priority": round(priority, 4),
            })
    return sorted(rows, key=lambda row: (-row["priority"], row["item"], row["action"]))[:12]


RECONSTRUCTION_TARGET_TERMS = {
    "A_assumption_graph_memory": ["Assumption Graph Memory", "Assumption Graph", "HippoRAG"],
    "B_hypothesis_generator": ["Hypothesis Generator", "新假设生成", "systematic residual"],
    "C_world_model_simulator": ["World Model / Simulator", "世界模型蒸馏", "廉价验证器"],
    "D_verifier_stack": ["Verifier Stack", "POPPER", "falsification"],
    "E_residual_analyzer": ["Residual Analyzer", "residual taxonomy", "失败归因"],
    "F_metaproductivity_selector": ["Metaproductivity", "clade", "HGM"],
    "G_formal_alignment_layer": ["Formal Alignment Layer", "范畴论", "信息几何"],
    "recursive_execution_loop": ["递归执行循环", "recursive", "多条候选假设轨迹"],
    "assumption_bench_evaluation": ["评价体系", "AssumptionBench", "answer win-rate"],
}


def _reconstruction_reference(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"exists": False, "target_count": len(RECONSTRUCTION_TARGET_TERMS), "matched_target_count": 0}
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "target_count": len(RECONSTRUCTION_TARGET_TERMS),
            "matched_target_count": 0,
        }
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    term_hits = {}
    for target, terms in RECONSTRUCTION_TARGET_TERMS.items():
        hits = [term for term in terms if term.lower() in lowered]
        term_hits[target] = {
            "matched": bool(hits),
            "matched_terms": hits,
            "required_any": terms,
        }
    return {
        "exists": True,
        "path": str(path),
        "byte_count": len(text.encode("utf-8")),
        "target_count": len(term_hits),
        "matched_target_count": sum(1 for row in term_hits.values() if row["matched"]),
        "term_hits": term_hits,
    }


RECONSTRUCTION_CEILINGS = {
    "A_assumption_graph_memory": (0.88, 0.78),
    "B_hypothesis_generator": (0.80, 0.70),
    "C_world_model_simulator": (0.82, 0.64),
    "D_verifier_stack": (0.82, 0.74),
    "E_residual_analyzer": (0.82, 0.74),
    "F_metaproductivity_selector": (0.80, 0.70),
    "G_formal_alignment_layer": (0.76, 0.65),
    "recursive_execution_loop": (0.85, 0.76),
    "assumption_bench_evaluation": (0.88, 0.82),
}


def _apply_reconstruction_ceiling(item: ProgressItem) -> ProgressItem:
    max_structure, max_behavior = RECONSTRUCTION_CEILINGS.get(item.key, (1.0, 1.0))
    gaps = list(item.remaining_gaps)
    if item.structure_score > max_structure or item.behavior_score > max_behavior:
        gaps.append(
            "Score is capped because reconstruction.md targets full mature behavior, not only the existence of passing local artifacts."
        )
    return ProgressItem(
        key=item.key,
        target=item.target,
        structure_score=min(item.structure_score, max_structure),
        behavior_score=min(item.behavior_score, max_behavior),
        evidence={**item.evidence, "reconstruction_ceiling": {"structure": max_structure, "behavior": max_behavior}},
        remaining_gaps=gaps,
        next_actions=item.next_actions,
    )


def _graph_stats(graph_dir: Path) -> dict[str, Any]:
    node_types: Counter[str] = Counter()
    edge_types: Counter[str] = Counter()
    positive_meta = 0
    node_count = 0
    edge_count = 0
    trial_count = 0
    nodes_path = graph_dir / "nodes.jsonl"
    if nodes_path.exists():
        for row in _iter_jsonl(nodes_path):
            node_count += 1
            node_types[str(row.get("type", ""))] += 1
            if float(row.get("metaproductivity") or 0.0) > 0.0:
                positive_meta += 1
    edges_path = graph_dir / "edges.jsonl"
    if edges_path.exists():
        for row in _iter_jsonl(edges_path):
            edge_count += 1
            edge_types[str(row.get("type", ""))] += 1
    trials_path = graph_dir / "trials.jsonl"
    if trials_path.exists():
        trial_count = sum(1 for _ in _iter_jsonl(trials_path))
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "trial_count": trial_count,
        "node_type_count": len(node_types),
        "edge_type_count": len(edge_types),
        "positive_metaproductivity_nodes": positive_meta,
    }


def _iter_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def _status(structure: float, behavior: float) -> str:
    if structure >= 0.9 and behavior >= 0.85:
        return "mature"
    if structure >= 0.8 and behavior >= 0.7:
        return "operational"
    if structure >= 0.65 and behavior >= 0.45:
        return "partial"
    return "early"


def _score_brier(value: Any, *, threshold: float) -> float:
    if value is None:
        return 0.0
    return _cap(1.0 - min(float(value) / threshold, 1.0))


def _avg(values: list[float]) -> float:
    return _mean(values)


def _mean(values) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _cap(value: Any) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str | Path | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else root / p


def _display_path(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--performance-payload", required=True)
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--reconstruction", default="reconstruction/md/reconstruction.md")
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    payload = build_reconstruction_progress_payload(
        root=root,
        performance_payload=_load_json(_resolve(root, args.performance_payload)),
        graph_dir=_resolve(root, args.graph_dir),
        reconstruction_path=_resolve(root, args.reconstruction),
        eval_id=args.eval_id,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
