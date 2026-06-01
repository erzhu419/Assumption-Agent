"""Performance validation for reconstruction gap closures.

This runner evaluates the reconstruction mechanisms added after reconstruction:

1. component manifest logging
2. cheap world model / simulator
3. multi-path trajectory search
4. recursive daemon execute/read/resume loop
5. residual clustering -> hypothesis synthesis
6. finite formal-mapping information geometry and safe deduplication
7. harness artifact observer coverage
8. unified verifier stack
9. recursive runner closure audit
10. evolution context / harness responsibility gate
11. assumption lifecycle capability scoreboard
12. runtime memory surfaces in graph memory
13. first-party runtime tracing for live LLM/retrieval calls
14. trace-to-outcome datasets for world-model/residual training
15. trace outcome model for route/component policy calibration
16. trace policy proposals for recursive verifier intake
17. trace policy preflight for fresh-ablation readiness
18. reconstruction progress audit against reconstruction.md

The validation uses existing real artifacts where available and deterministic
positive controls where the mechanism needs a safe graph-mutation sandbox.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .assumption_bench import build_assumption_bench_payload
from .falsification import build_falsification_payload
from .evolution_context import build_evolution_context_payload
from .formal_mapping import (
    build_categorical_info_geometry_payload,
    build_formal_dedup_payload,
    build_formal_mapping_payload,
    build_formal_search_eval_payload,
    build_formal_transfer_eval_payload,
)
from .graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from .harness_observer import build_harness_observer_payload
from .manifest_logger import build_component_manifest_payload, events_from_run_logs
from .memory_surfaces import build_memory_surface_payload
from .recursive_audit import build_recursive_audit_payload
from .recursive_daemon import build_recursive_daemon_payload
from .recursive_executor import JudgmentSet
from .reconstruction_progress import build_reconstruction_progress_payload
from .recursive_runner import build_recursive_assumption_run
from .residual_clusterer import build_residual_cluster_payload
from .runtime_trace import RuntimeTraceRecorder
from .surface_hypotheses import build_surface_hypothesis_payload
from .selector import build_metaproductivity_benchmark_payload
from .trajectory_search import build_trajectory_search_payload
from .trace_dataset import build_trace_dataset_collection_payload, build_trace_dataset_payload
from .trace_outcome_model import build_trace_outcome_model_payload, build_trace_policy_proposal_payload
from .verifier_stack import build_verifier_stack_payload
from .world_model import build_world_model_payload, train_world_model_calibration
from .schema import AssumptionNode, AssumptionType


DEFAULT_ARTIFACT_DIR = Path("phase four/assumption_graph")


def build_performance_validation_payload(
    *,
    root: Path,
    graph_dir: Path,
    eval_id: str,
) -> dict:
    timings: dict[str, float] = {}

    start = time.perf_counter()
    world = _validate_world_model(root=root, graph_dir=graph_dir)
    timings["world_model_sec"] = _elapsed(start)

    start = time.perf_counter()
    trajectory = _validate_trajectory_search(root=root, world_model_payload=world["post_acceptance_payload"])
    timings["trajectory_search_sec"] = _elapsed(start)

    start = time.perf_counter()
    metaproductivity = _validate_metaproductivity_benchmark(graph_dir=graph_dir)
    timings["metaproductivity_benchmark_sec"] = _elapsed(start)

    start = time.perf_counter()
    verifier = _validate_verifier_stack(root=root, world_model_payload=world["post_acceptance_payload"])
    timings["verifier_stack_sec"] = _elapsed(start)

    start = time.perf_counter()
    daemon = _validate_recursive_daemon(root=root, graph_dir=graph_dir)
    timings["recursive_daemon_sec"] = _elapsed(start)

    start = time.perf_counter()
    recursive_audit = _validate_recursive_audit(root=root, graph_dir=graph_dir)
    timings["recursive_audit_sec"] = _elapsed(start)

    start = time.perf_counter()
    manifests = _validate_manifest_logger(root=root)
    timings["manifest_logger_sec"] = _elapsed(start)

    start = time.perf_counter()
    runtime_trace = _validate_runtime_trace()
    timings["runtime_trace_sec"] = _elapsed(start)

    start = time.perf_counter()
    trace_dataset = _validate_trace_dataset(root=root)
    timings["trace_dataset_sec"] = _elapsed(start)

    start = time.perf_counter()
    trace_outcome_model = _validate_trace_outcome_model(root=root)
    timings["trace_outcome_model_sec"] = _elapsed(start)

    start = time.perf_counter()
    trace_policy_proposals = _validate_trace_policy_proposals(root=root, graph_dir=graph_dir)
    timings["trace_policy_proposals_sec"] = _elapsed(start)

    start = time.perf_counter()
    trace_policy_preflight = _validate_trace_policy_preflight(root=root)
    timings["trace_policy_preflight_sec"] = _elapsed(start)

    start = time.perf_counter()
    harness = _validate_harness_observer(root=root, graph_dir=graph_dir)
    timings["harness_observer_sec"] = _elapsed(start)

    start = time.perf_counter()
    residuals = _validate_residual_clusterer(graph_dir=graph_dir)
    timings["residual_clusterer_sec"] = _elapsed(start)

    start = time.perf_counter()
    formal = _validate_formal_metrics(root=root, graph_dir=graph_dir)
    timings["formal_metrics_sec"] = _elapsed(start)

    sections = {
        "world_model": _strip_payload(world),
        "trajectory_search": trajectory,
        "metaproductivity_benchmark": metaproductivity,
        "verifier_stack": verifier,
        "recursive_daemon": daemon,
        "recursive_audit": recursive_audit,
        "manifest_logger": manifests,
        "runtime_trace": runtime_trace,
        "trace_dataset": trace_dataset,
        "trace_outcome_model": trace_outcome_model,
        "trace_policy_proposals": trace_policy_proposals,
        "trace_policy_preflight": trace_policy_preflight,
        "harness_observer": harness,
        "residual_clusterer": residuals,
        "formal_metrics": formal,
    }
    start = time.perf_counter()
    sections["surface_hypothesis_generator"] = _validate_surface_hypothesis_generator(
        graph_dir=graph_dir,
        sections=sections,
    )
    timings["surface_hypothesis_generator_sec"] = _elapsed(start)
    start = time.perf_counter()
    sections["evolution_context"] = _validate_evolution_context(sections=sections)
    timings["evolution_context_sec"] = _elapsed(start)
    start = time.perf_counter()
    sections["memory_surfaces"] = _validate_memory_surfaces(root=root, graph_dir=graph_dir, sections=sections)
    timings["memory_surfaces_sec"] = _elapsed(start)
    start = time.perf_counter()
    sections["assumption_bench"] = _validate_assumption_bench(sections=sections, graph_dir=graph_dir)
    timings["assumption_bench_sec"] = _elapsed(start)
    start = time.perf_counter()
    sections["reconstruction_progress"] = _validate_reconstruction_progress(
        root=root,
        graph_dir=graph_dir,
        eval_id=eval_id,
        sections=sections,
    )
    timings["reconstruction_progress_sec"] = _elapsed(start)
    return {
        "eval_id": eval_id,
        "source": {
            "root": ".",
            "graph_dir": _display_path(root, graph_dir),
        },
        "timings": timings,
        "overall_pass": all(section.get("pass", False) for section in sections.values()),
        "sections": sections,
    }


def format_performance_report(payload: dict) -> str:
    sections = payload["sections"]
    lines = [
        f"# Reconstruction Gap Performance Validation: {payload['eval_id']}",
        "",
        f"Overall: {'PASS' if payload['overall_pass'] else 'FAIL'}",
        "",
        "## Summary",
        "",
        "| Gap | Result | Key Metric |",
        "| --- | --- | --- |",
    ]
    for name, section in sections.items():
        lines.append(
            f"| {name} | {'PASS' if section.get('pass') else 'FAIL'} | {_key_metric(name, section)} |"
        )
    lines.extend(["", "## Details", ""])
    for name, section in sections.items():
        lines.extend([f"### {name}", ""])
        for key, value in section.items():
            if key in {"pass", "notes"}:
                continue
            lines.append(f"- `{key}`: {json.dumps(value, ensure_ascii=False, sort_keys=True)}")
        if section.get("notes"):
            for note in section["notes"]:
                lines.append(f"- note: {note}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _validate_world_model(*, root: Path, graph_dir: Path) -> dict:
    bundle = _combined_candidate_bundle(root)
    store = JsonlGraphStore(graph_dir)
    pre_raw = build_world_model_payload(
        store=store,
        proposal_payload=bundle["proposal_payload"],
        preflight_payload=bundle["preflight_payload"],
        falsification_payload=bundle["falsification_payload"],
        regression_predictions=bundle["regression_predictions"],
        formal_mapping_gate_payload=bundle["formal_gate_payload"],
        eval_id="perf_world_pre_acceptance_raw",
    )
    calibration_model = train_world_model_calibration(
        prediction_payload=pre_raw,
        acceptance_payload=bundle["acceptance_payload"],
        eval_id="perf_world_calibration",
    )
    pre = build_world_model_payload(
        store=store,
        proposal_payload=bundle["proposal_payload"],
        preflight_payload=bundle["preflight_payload"],
        falsification_payload=bundle["falsification_payload"],
        regression_predictions=bundle["regression_predictions"],
        formal_mapping_gate_payload=bundle["formal_gate_payload"],
        calibration_payload=calibration_model,
        eval_id="perf_world_pre_acceptance_calibrated",
    )
    post = build_world_model_payload(
        store=store,
        proposal_payload=bundle["proposal_payload"],
        preflight_payload=bundle["preflight_payload"],
        falsification_payload=bundle["falsification_payload"],
        acceptance_payload=bundle["acceptance_payload"],
        regression_predictions=bundle["regression_predictions"],
        formal_mapping_gate_payload=bundle["formal_gate_payload"],
        calibration_payload=calibration_model,
        eval_id="perf_world_post_acceptance",
    )
    labels = bundle["labels"]
    raw_pre_metrics = _rank_metrics(pre_raw["predictions"], labels)
    pre_metrics = _rank_metrics(pre["predictions"], labels)
    post_metrics = _rank_metrics(post["predictions"], labels)
    calibration = post.get("calibration", {})
    passed = (
        pre_metrics["labeled_count"] >= 16
        and pre_metrics["accepted_count"] >= 2
        and pre_metrics["auc"] is not None
        and pre_metrics["auc"] >= 0.85
        and pre_metrics["accepted_recall_at_k"] >= 1.0
        and calibration_model["calibrated_metrics"]["brier_score"] < calibration_model["raw_metrics"]["brier_score"]
        and calibration_model["leave_one_out_calibrated_metrics"]["brier_score"] < calibration_model["raw_metrics"]["brier_score"]
        and (calibration.get("brier_score") is not None and calibration["brier_score"] <= 0.08)
    )
    return {
        "pass": passed,
        "label_counts": dict(Counter(labels.values())),
        "matched_label_count": pre_metrics["labeled_count"],
        "unmatched_label_count": len(labels) - pre_metrics["labeled_count"],
        "raw_pre_acceptance": raw_pre_metrics,
        "pre_acceptance": pre_metrics,
        "post_acceptance": post_metrics,
        "post_calibration": calibration,
        "trained_calibration": {
            key: value
            for key, value in calibration_model.items()
            if key != "training_rows"
        },
        "prediction_count": pre["prediction_count"],
        "post_acceptance_payload": post,
        "notes": [
            "pre_acceptance excludes candidate acceptance labels to avoid leakage",
            "post_acceptance validates calibration/logging after real judgments are attached",
        ],
    }


def _validate_trajectory_search(*, root: Path, world_model_payload: dict) -> dict:
    recursive = {
        "eval_id": "perf_trajectory_frontier",
        "next_actions": [],
    }
    for name in [
        "recursive_positive_ms_bridge_runner.json",
        "recursive_positive_se_hard_policy_runner.json",
        "recursive_runner_phase2_v20_gpt55_21_50_resumed.json",
    ]:
        payload = _load_json(root / DEFAULT_ARTIFACT_DIR / name)
        recursive["next_actions"].extend(payload.get("next_actions", []))
    payload = build_trajectory_search_payload(
        recursive_payload=recursive,
        world_model_payload=world_model_payload,
        eval_id="perf_trajectory_search",
        beam_width=20,
        max_paths_per_candidate=4,
    )
    labels = _combined_candidate_bundle(root)["labels"]
    trajectories_by_proposal = defaultdict(list)
    for row in payload["trajectories"]:
        trajectories_by_proposal[row["proposal_id"]].append(row)
    top_hits = []
    for proposal_id, rows in trajectories_by_proposal.items():
        label = labels.get(proposal_id)
        if label not in {"accept", "reject"}:
            continue
        top = sorted(rows, key=lambda r: -float(r["score"]))[0]
        hit = (
            top["path_type"] == "promote_after_verification"
            if label == "accept"
            else top["path_type"] != "promote_after_verification"
        )
        top_hits.append(hit)
    multi_path = sum(1 for rows in trajectories_by_proposal.values() if len(rows) >= 2)
    proposal_count = len(trajectories_by_proposal)
    top_hit_rate = sum(top_hits) / len(top_hits) if top_hits else 0.0
    multi_path_rate = multi_path / proposal_count if proposal_count else 0.0
    return {
        "pass": proposal_count >= 8 and multi_path_rate >= 0.7 and top_hit_rate >= 0.85,
        "frontier_actions": len(recursive["next_actions"]),
        "trajectory_count": payload["trajectory_count"],
        "proposal_count": proposal_count,
        "multi_path_rate": round(multi_path_rate, 4),
        "top_path_label_hit_rate": round(top_hit_rate, 4),
        "path_type_counts": payload["path_type_counts"],
        "selected_path_types": Counter(row["path_type"] for row in payload["selected"]),
    }


def _validate_metaproductivity_benchmark(*, graph_dir: Path) -> dict:
    payload = build_metaproductivity_benchmark_payload(
        SimpleAssumptionGraph(JsonlGraphStore(graph_dir)),
        eval_id="perf_metaproductivity_benchmark",
    )
    live = payload.get("live_probe", {})
    mean_acp = live.get("mean_acp_top_clade_metaproductivity")
    mean_immediate = live.get("mean_immediate_top_clade_metaproductivity")
    return {
        "pass": (
            payload.get("pass", False)
            and payload.get("positive_control", {}).get("pass", False)
            and mean_acp is not None
            and mean_immediate is not None
            and mean_acp > mean_immediate
        ),
        "query_count": payload["query_count"],
        "positive_control": payload["positive_control"],
        "mean_acp_top_clade_metaproductivity": mean_acp,
        "mean_immediate_top_clade_metaproductivity": mean_immediate,
        "distinct_acp_top_count": live.get("distinct_acp_top_count"),
        "distinct_immediate_top_count": live.get("distinct_immediate_top_count"),
        "live_probe": live,
    }


def _validate_verifier_stack(*, root: Path, world_model_payload: dict) -> dict:
    bundle = _combined_candidate_bundle(root)
    payload = build_verifier_stack_payload(
        proposal_payload=bundle["proposal_payload"],
        preflight_payload=bundle["preflight_payload"],
        world_model_payload=world_model_payload,
        falsification_payload=bundle["falsification_payload"],
        acceptance_payload=bundle["acceptance_payload"],
        formal_mapping_gate_payload=bundle["formal_gate_payload"],
        eval_id="perf_verifier_stack",
    )
    accepted = [
        row for row in payload["summaries"]
        if row["verdict"] == "accepted_for_gated_apply"
    ]
    rejected = [
        row for row in payload["summaries"]
        if row["verdict"] in {"rejected_control_harm", "rejected_weak_benefit"}
    ]
    staged = [
        stage
        for row in payload["summaries"]
        for stage in row.get("stages", [])
    ]
    stage_status_counts = dict(Counter(f"{stage['tier']}:{stage['status']}" for stage in staged))
    falsification_experiments = [
        (row, experiment)
        for row in payload["summaries"]
        for stage in row.get("stages", [])
        if stage.get("tier") == "V3"
        for experiment in stage.get("evidence", {}).get("experiments", [])
    ]
    experiment_status_counts = dict(Counter(
        experiment.get("status") for _, experiment in falsification_experiments
    ))
    experiment_name_counts = dict(Counter(
        experiment.get("name") for _, experiment in falsification_experiments
    ))
    protocol_candidate_count = len({
        row["proposal_id"]
        for row, experiment in falsification_experiments
        if experiment.get("name") == "trigger_benefit_sequential"
    })

    def has_experiment(row: dict, name: str, statuses: set[str]) -> bool:
        for stage in row.get("stages", []):
            if stage.get("tier") != "V3":
                continue
            for experiment in stage.get("evidence", {}).get("experiments", []):
                if experiment.get("name") == name and experiment.get("status") in statuses:
                    return True
        return False

    accepted_protocol_ok = all(
        has_experiment(row, "trigger_benefit_sequential", {"passed"})
        and has_experiment(row, "control_harm_sequential", {"passed"})
        and has_experiment(row, "fresh_cross_judge_replay", {"passed"})
        for row in accepted
    )
    rejected_protocol_ok = all(
        (
            row["verdict"] == "rejected_weak_benefit"
            and has_experiment(row, "trigger_benefit_sequential", {"failed"})
        )
        or (
            row["verdict"] == "rejected_control_harm"
            and has_experiment(row, "control_harm_sequential", {"failed"})
        )
        for row in rejected
    )
    passed = (
        payload["proposal_count"] >= 16
        and len(accepted) >= 2
        and len(rejected) >= 10
        and all(row["next_action"] == "apply_accepted_candidate_if_requested" for row in accepted)
        and stage_status_counts.get("V4:pass", 0) >= 2
        and stage_status_counts.get("V4:fail", 0) >= 10
        and protocol_candidate_count >= 16
        and accepted_protocol_ok
        and rejected_protocol_ok
    )
    return {
        "pass": passed,
        "proposal_count": payload["proposal_count"],
        "verdict_counts": payload["verdict_counts"],
        "confidence_counts": payload["confidence_counts"],
        "next_action_counts": payload["next_action_counts"],
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "stage_status_counts": stage_status_counts,
        "falsification_experiment_count": len(falsification_experiments),
        "falsification_protocol_candidate_count": protocol_candidate_count,
        "falsification_experiment_status_counts": experiment_status_counts,
        "falsification_experiment_name_counts": experiment_name_counts,
        "accepted_protocol_ok": accepted_protocol_ok,
        "rejected_protocol_ok": rejected_protocol_ok,
    }


def _validate_recursive_daemon(*, root: Path, graph_dir: Path) -> dict:
    cases = [
        ("ms_bridge", "recursive_positive_ms_bridge_runner.json", "recursive_positive_ms_bridge_evolution.json", "recursive_positive_ms_bridge_judgment_bundle.json"),
        ("se_hard_policy", "recursive_positive_se_hard_policy_runner.json", "recursive_positive_se_hard_policy_evolution.json", "recursive_positive_se_hard_policy_judgment_bundle.json"),
    ]
    results = []
    with tempfile.TemporaryDirectory() as td:
        tmp_graph = Path(td) / "graph"
        _copy_graph_store(graph_dir, tmp_graph)
        for label, runner_name, evolution_name, bundle_name in cases:
            recursive_payload = _load_json(root / DEFAULT_ARTIFACT_DIR / runner_name)
            evolution_payload = _load_json(root / DEFAULT_ARTIFACT_DIR / evolution_name)
            judgment_sets = _judgment_sets_from_bundle(root, _load_json(root / DEFAULT_ARTIFACT_DIR / bundle_name))
            dry = build_recursive_daemon_payload(
                root=root,
                graph_dir=tmp_graph,
                recursive_payload=recursive_payload,
                evolution_payload=evolution_payload,
                eval_id=f"perf_daemon_{label}_dry",
                judgment_sets=judgment_sets,
                apply_accepted=False,
                writeback_manifests=False,
            )
            before_nodes = set(JsonlGraphStore(tmp_graph).nodes)
            applied = build_recursive_daemon_payload(
                root=root,
                graph_dir=tmp_graph,
                recursive_payload=recursive_payload,
                evolution_payload=evolution_payload,
                eval_id=f"perf_daemon_{label}_apply",
                judgment_sets=judgment_sets,
                apply_accepted=True,
                writeback_manifests=True,
            )
            after_store = JsonlGraphStore(tmp_graph)
            applied_ids = applied.get("applied_candidate_node_ids", [])
            results.append({
                "case": label,
                "dry_applied_count": len(dry.get("applied_candidate_node_ids", [])),
                "dry_mutated": set(JsonlGraphStore(tmp_graph).nodes) != before_nodes and not applied_ids,
                "accepted_counts": applied["iterations"][0]["candidate_acceptance_counts"],
                "applied_candidate_node_ids": applied_ids,
                "applied_nodes_present": all(node_id in after_store.nodes for node_id in applied_ids),
                "manifest_count": applied["manifest_count"],
            })
    passed = all(
        row["dry_applied_count"] == 0
        and row["accepted_counts"].get("accept") == 1
        and len(row["applied_candidate_node_ids"]) == 1
        and row["applied_nodes_present"]
        and row["manifest_count"] >= 2
        for row in results
    )
    return {
        "pass": passed,
        "case_count": len(results),
        "accepted_apply_count": sum(len(r["applied_candidate_node_ids"]) for r in results),
        "results": results,
    }


def _validate_recursive_audit(*, root: Path, graph_dir: Path) -> dict:
    artifact_dir = root / DEFAULT_ARTIFACT_DIR
    cycle = _load_json(artifact_dir / "evolution_cycle_dryrun_phase2_v20_gpt55_21_50.json")
    positive = _load_json(artifact_dir / "recursive_positive_ms_bridge_evolution.json")
    positive_acceptance = _load_json(artifact_dir / "recursive_positive_ms_bridge_acceptance.json")
    dry_payload = build_recursive_assumption_run(
        graph_dir=graph_dir,
        problem="Audit dry recursive graph self-evolution frontier.",
        goal="Verify that open hypothesis frames expose child evidence gaps.",
        eval_id="perf_recursive_audit_dry",
        problem_id="perf_recursive_audit_dry",
        evolution_payload=cycle,
        max_children=4,
        max_depth=3,
    )
    accepted_payload = build_recursive_assumption_run(
        graph_dir=graph_dir,
        problem="Audit accepted recursive graph self-evolution frontier.",
        goal="Verify that accepted child evidence returns a parent update.",
        eval_id="perf_recursive_audit_accepted",
        problem_id="perf_recursive_audit_accepted",
        evolution_payload=positive,
        acceptance_payload=positive_acceptance,
        max_children=2,
        max_depth=3,
    )
    audits = [
        build_recursive_audit_payload(
            recursive_payload=dry_payload,
            eval_id="perf_recursive_audit_dry",
        ),
        build_recursive_audit_payload(
            recursive_payload=accepted_payload,
            eval_id="perf_recursive_audit_accepted",
        ),
    ]
    critical = sum(audit.get("issue_counts", {}).get("critical", 0) for audit in audits)
    warning = sum(audit.get("issue_counts", {}).get("warning", 0) for audit in audits)
    min_score = min(audit["closure_score"] for audit in audits)
    return {
        "pass": all(audit["pass"] for audit in audits) and critical == 0,
        "case_count": len(audits),
        "frame_count": sum(audit["frame_count"] for audit in audits),
        "actionable_count": sum(audit["actionable_count"] for audit in audits),
        "critical_issue_count": critical,
        "warning_issue_count": warning,
        "min_closure_score": min_score,
        "case_summaries": [
            {
                "eval_id": audit["eval_id"],
                "pass": audit["pass"],
                "frame_count": audit["frame_count"],
                "actionable_count": audit["actionable_count"],
                "closure_score": audit["closure_score"],
                "issue_counts": audit["issue_counts"],
            }
            for audit in audits
        ],
    }


def _validate_evolution_context(*, sections: dict[str, dict]) -> dict:
    dry = build_evolution_context_payload(
        eval_id="perf_evolution_context_dry",
        objective="Govern recursive graph self-evolution with explicit harness responsibilities.",
        sections=sections,
    )
    apply_allowed = build_evolution_context_payload(
        eval_id="perf_evolution_context_apply",
        objective="Allow bounded gated apply only when accepted candidates and harness checks pass.",
        sections=sections,
        mode={"apply_accepted": True},
        permissions={"allow_apply_accepted": True, "max_apply_candidates": 2},
    )
    blocked = build_evolution_context_payload(
        eval_id="perf_evolution_context_blocked",
        objective="Confirm apply is blocked without explicit permission.",
        sections=sections,
        mode={"apply_accepted": True},
    )
    dry_passes = dry["policy_decision"] == "ready_for_manual_apply"
    apply_passes = apply_allowed["policy_decision"] == "gated_apply_allowed"
    blocked_passes = (
        blocked["policy_decision"] == "blocked_by_permissions"
        and blocked["permission_violations"]
    )
    responsibilities_pass = dry["responsibility_status_counts"].get("pass", 0) >= 9
    return {
        "pass": dry_passes and apply_passes and blocked_passes and responsibilities_pass,
        "responsibility_count": dry["responsibility_count"],
        "responsibility_status_counts": dry["responsibility_status_counts"],
        "dry_policy_decision": dry["policy_decision"],
        "apply_policy_decision": apply_allowed["policy_decision"],
        "blocked_policy_decision": blocked["policy_decision"],
        "blocked_violation_count": len(blocked["permission_violations"]),
        "accepted_candidate_count": dry["accepted_candidate_count"],
        "actionable_frontier_count": dry["actionable_frontier_count"],
        "procedure_update_count": len(dry["procedure_updates"]),
        "procedure_update_ids": [row["id"] for row in dry["procedure_updates"]],
    }


def _validate_assumption_bench(*, sections: dict[str, dict], graph_dir: Path) -> dict:
    payload = build_assumption_bench_payload(
        eval_id="perf_assumption_bench",
        sections=sections,
        graph_dir=graph_dir,
    )
    return {
        "pass": payload["pass"],
        "overall_score": payload["overall_score"],
        "min_score": payload["min_score"],
        "capability_count": payload["capability_count"],
        "passed_capability_count": payload["passed_capability_count"],
        "failed_capabilities": payload["failed_capabilities"],
        "score_by_capability": {
            row["name"]: row["score"]
            for row in payload["scores"]
        },
    }


def _validate_reconstruction_progress(
    *,
    root: Path,
    graph_dir: Path,
    eval_id: str,
    sections: dict[str, dict],
) -> dict:
    payload = build_reconstruction_progress_payload(
        root=root,
        performance_payload={"eval_id": eval_id, "sections": sections},
        graph_dir=graph_dir,
        reconstruction_path=root / "reconstruction/md/reconstruction.md",
        eval_id="perf_reconstruction_progress",
    )
    closure = payload["closure"]
    return {
        "pass": payload["overall_pass"],
        "structure_percent": closure["structure_percent"],
        "behavior_percent": closure["behavior_percent"],
        "weighted_percent": closure["weighted_percent"],
        "completed_item_count": closure["completed_item_count"],
        "item_count": closure["item_count"],
        "status_counts": dict(Counter(row["status"] for row in payload["items"])),
        "lowest_behavior_items": [
            {
                "key": row["key"],
                "behavior_score": row["behavior_score"],
                "structure_score": row["structure_score"],
                "status": row["status"],
            }
            for row in sorted(payload["items"], key=lambda row: (row["behavior_score"], row["structure_score"]))[:3]
        ],
        "top_next_actions": payload["next_actions_ranked"][:5],
    }


def _validate_memory_surfaces(*, root: Path, graph_dir: Path, sections: dict[str, dict]) -> dict:
    with tempfile.TemporaryDirectory() as td:
        tmp_graph = Path(td) / "graph"
        _copy_graph_store(graph_dir, tmp_graph)
        payload = build_memory_surface_payload(
            graph_dir=tmp_graph,
            eval_id="perf_memory_surfaces",
            performance_payload={"eval_id": "perf_partial_sections", "sections": sections},
            writeback=True,
        )
    after = payload["after_graph"]
    return {
        "pass": payload["memory_transfer_ready"],
        "surface_count": payload["surface_count"],
        "edge_count": payload["edge_count"],
        "new_node_count": payload["new_node_count"],
        "new_edge_count": payload["new_edge_count"],
        "before_node_type_count": payload["before_graph"]["node_type_count"],
        "after_node_type_count": after["node_type_count"],
        "before_edge_type_count": payload["before_graph"]["edge_type_count"],
        "after_edge_type_count": after["edge_type_count"],
        "node_type_counts": after["node_type_counts"],
        "edge_type_counts": after["edge_type_counts"],
    }


def _validate_manifest_logger(*, root: Path) -> dict:
    events = [
        {
            "event_type": kind,
            "problem_id": f"manifest_perf_{idx}",
            "component": f"component_{kind}",
            "assumption": "Every component event should become a redacted TrialManifest.",
            "why_selected": "Performance validation exercises high-volume manifest logging.",
            "expected_effect": "Append a manifest without leaking secrets.",
            "observed_effect": "event observed",
            "artifacts": {"request": f"secret_token=redaction-probe-{idx}", "payload_size": idx},
            "metadata": {"model": "validation-model", "iteration": idx},
        }
        for idx, kind in enumerate(["llm_call", "retrieval", "judge_call", "tool_use", "simulator_rollout"] * 20)
    ]
    real_log_paths = [
        root / "phase four/assumption_graph/recursive_scoped_judge_run_gpt55_21_50.log",
        root / "phase four/assumption_graph/recursive_scoped_ablation_run_gpt55_21_50.log",
        root / "phase four/assumption_graph/candidate_ablation_run_phase2_v20_gpt54mini_21_50.log",
        root / "phase four/assumption_graph/candidate_ablation_run_phase2_v20_gpt55_21_50.log",
        root / "phase six/autonomous/exp80_run.log",
    ]
    real_events = events_from_run_logs(
        root=root,
        log_paths=real_log_paths,
        max_events_per_file=20,
    )
    with tempfile.TemporaryDirectory() as td:
        store = JsonlGraphStore(td)
        start = time.perf_counter()
        payload = build_component_manifest_payload(
            eval_id="perf_manifest_logger",
            events=[*events, *real_events],
            store=store,
            writeback=True,
        )
        elapsed = _elapsed(start)
        text = json.dumps(payload, ensure_ascii=False)
        leak = "redaction-probe-" in text
        written = len(JsonlGraphStore(td).trials)
    total_events = len(events) + len(real_events)
    throughput = total_events / elapsed if elapsed else float("inf")
    return {
        "pass": written == total_events and len(real_events) >= 5 and not leak and throughput >= 100.0,
        "event_count": total_events,
        "synthetic_event_count": len(events),
        "real_log_event_count": len(real_events),
        "real_log_paths": [
            _display_path(root, path)
            for path in real_log_paths
            if path.exists()
        ],
        "written_trials": written,
        "secret_leak_detected": leak,
        "throughput_events_per_sec": round(throughput, 2),
        "event_counts": payload["event_counts"],
    }


def _validate_runtime_trace() -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        graph_dir = root / "graph"
        recorder = RuntimeTraceRecorder(
            eval_id="perf_runtime_trace",
            events_out=root / "events.jsonl",
            summary_out=root / "summary.json",
            graph_dir=graph_dir,
            writeback=True,
        )
        recorder.record_retrieval(
            problem_id="runtime_trace_retrieval",
            component="phase2_assumption_graph_retrieval",
            assumption="Live graph retrieval should emit a first-party assumption manifest.",
            expected_effect="Make activated assumption ids, node types, and policy notes available without parsing logs.",
            activated_assumption_ids=["strategy_S01", "surface_7f67a660e3c7"],
            artifacts={
                "node_types": ["method", "verifier"],
                "policy_notes": ["runtime trace positive control"],
                "query": "redaction test api_key=runtime-trace-probe",
            },
        )
        recorder.record_llm_call(
            problem_id="runtime_trace_draft",
            component="phase2_turn1_draft",
            prompt_kind="execute_v20",
            assumption="A live draft call should be logged as a compact assumption-bearing LLM event.",
            expected_effect="Record call metadata and outcome shape without storing full prompts or secrets.",
            observed_effect="draft_chars=128",
            artifacts={"request": "secret_token=runtime-trace-secret", "max_tokens": 1100},
        )
        recorder.record(
            event_type="tool_use",
            problem_id="runtime_trace_cache",
            component="phase2_cache_write",
            assumption="Cache writes are tool-use events that affect later evaluation.",
            why_selected="The runner stores intermediate artifacts used by future passes.",
            expected_effect="Persist bounded metadata about cache side effects.",
            observed_effect="cache_keys=3",
            artifacts={"path": "phase two/analysis/cache/answers/example.json"},
        )
        payload = recorder.flush()
        event_text = (root / "events.jsonl").read_text(encoding="utf-8")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        leak = "runtime-trace-secret" in event_text or "runtime-trace-probe" in event_text
        written = len(JsonlGraphStore(graph_dir).trials)
        return {
            "pass": (
                payload["event_count"] == 3
                and written == 3
                and summary["event_count"] == 3
                and not leak
                and set(payload["event_counts"]) >= {"retrieval", "llm_call", "tool_use"}
            ),
            "event_count": payload["event_count"],
            "event_counts": payload["event_counts"],
            "written_trials": written,
            "events_out_written": (root / "events.jsonl").exists(),
            "summary_out_written": (root / "summary.json").exists(),
            "secret_leak_detected": leak,
        }


def _validate_trace_dataset(*, root: Path) -> dict:
    with tempfile.TemporaryDirectory() as td:
        temp_root = Path(td)
        sample_path = temp_root / "sample.json"
        meta_path = temp_root / "meta.json"
        judgments_path = temp_root / "candidate_vs_baseline.json"
        events_path = temp_root / "events.jsonl"
        sample_path.write_text(json.dumps([
            {
                "problem_id": "trace_ds_win",
                "domain": "software_engineering",
                "difficulty": "hard",
                "coverage_tags": ["S01"],
            },
            {
                "problem_id": "trace_ds_loss",
                "domain": "science",
                "difficulty": "medium",
                "coverage_tags": ["S12"],
            },
        ], ensure_ascii=False), encoding="utf-8")
        meta_path.write_text(json.dumps({
            "trace_ds_win": {"frame": "hybrid"},
            "trace_ds_loss": {"frame": "object", "bypass_route": "science_mechanism"},
        }, ensure_ascii=False), encoding="utf-8")
        judgments_path.write_text(json.dumps({
            "trace_ds_win": {
                "winner": "candidate",
                "score_a": 9,
                "score_b": 7,
                "a_was": "A",
                "reasoning": "candidate used the active assumption concretely",
            },
            "trace_ds_loss": {
                "winner": "baseline",
                "score_a": 6,
                "score_b": 8,
                "a_was": "A",
                "reasoning": "baseline gave a more concrete validation bridge",
            },
        }, ensure_ascii=False), encoding="utf-8")
        events = [
            {
                "event_type": "retrieval",
                "problem_id": "trace_ds_win",
                "component": "phase2_assumption_graph_retrieval",
                "assumption": "retrieval should activate useful graph context",
                "artifacts": {
                    "activated_assumption_ids": ["strategy_S01"],
                    "query": "api_key=trace-dataset-probe",
                },
            },
            {
                "event_type": "llm_call",
                "problem_id": "trace_ds_win",
                "component": "phase2_turn1_draft",
                "assumption": "draft call should use retrieved assumptions",
                "artifacts": {"prompt_kind": "execute_v20"},
            },
            {
                "event_type": "tool_use",
                "problem_id": "trace_ds_loss",
                "component": "phase2_cache_hit",
                "assumption": "cached bypass output is trace evidence",
                "artifacts": {
                    "bypass_route": "science_mechanism",
                    "request": "secret_token=trace-dataset-secret",
                },
            },
        ]
        events_path.write_text(
            "\n".join(json.dumps(event, ensure_ascii=False, sort_keys=True) for event in events) + "\n",
            encoding="utf-8",
        )
        payload = build_trace_dataset_payload(
            root=temp_root,
            sample_path=sample_path,
            meta_path=meta_path,
            judgments_path=judgments_path,
            trace_events_path=events_path,
            intervention_variant="candidate",
            baseline_variant="baseline",
            eval_id="perf_trace_dataset",
        )
        real_paths = [
            root / "phase four/assumption_graph/trace_dataset_ms_bridge_20260601.json",
            root / "phase four/assumption_graph/trace_dataset_ms_bridge_ms100_20260601.json",
            root / "phase four/assumption_graph/trace_dataset_ms_bridge_ms100_vs_v20_20260601.json",
        ]
        real_payloads = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in real_paths
            if path.exists()
        ]
        collection = build_trace_dataset_collection_payload(
            root=root,
            trace_dataset_payloads=real_payloads,
            eval_id="perf_trace_dataset_collection",
        ) if real_payloads else {}
        positive_control_pass = (
            payload["row_count"] == 2
            and payload["trainable_row_count"] == 2
            and payload["first_party_trace_count"] == 2
            and payload["traced_outcome_coverage"] == 1.0
            and payload["outcome_counts"] == {"loss": 1, "win": 1}
            and payload["residual_type_counts"].get("optimization") == 1
            and not payload["secret_leak_detected"]
        )
        collection_pass = (
            collection.get("dataset_count", 0) >= 3
            and collection.get("trainable_row_count", 0) >= 60
            and collection.get("first_party_trainable_row_count", 0) >= 9
            and collection.get("artifact_replay_trainable_row_count", 0) >= 50
            and collection.get("weighted_trainable_row_count", 0.0) >= 35.0
            and not collection.get("secret_leak_detected", True)
        )
        return {
            "pass": positive_control_pass and collection_pass,
            "row_count": collection.get("row_count", payload["row_count"]),
            "trainable_row_count": collection.get("trainable_row_count", payload["trainable_row_count"]),
            "weighted_trainable_row_count": collection.get("weighted_trainable_row_count", payload["trainable_row_count"]),
            "first_party_trace_count": collection.get("first_party_trace_count", payload["first_party_trace_count"]),
            "first_party_trainable_row_count": collection.get("first_party_trainable_row_count", payload["first_party_trace_count"]),
            "artifact_replay_count": collection.get("artifact_replay_count", payload["artifact_replay_count"]),
            "artifact_replay_trainable_row_count": collection.get("artifact_replay_trainable_row_count", 0),
            "missing_trace_count": collection.get("missing_trace_count", payload["missing_trace_count"]),
            "traced_outcome_coverage": payload["traced_outcome_coverage"],
            "assumption_id_coverage": payload["assumption_id_coverage"],
            "outcome_counts": collection.get("outcome_counts", payload["outcome_counts"]),
            "residual_type_counts": collection.get("residual_type_counts", payload["residual_type_counts"]),
            "event_counts": collection.get("event_counts", payload["event_counts"]),
            "source_eval_ids": collection.get("source", {}).get("source_eval_ids", []),
            "positive_control": {
                "pass": positive_control_pass,
                "row_count": payload["row_count"],
                "trainable_row_count": payload["trainable_row_count"],
                "first_party_trace_count": payload["first_party_trace_count"],
                "secret_leak_detected": payload["secret_leak_detected"],
            },
            "secret_leak_detected": payload["secret_leak_detected"] or collection.get("secret_leak_detected", False),
        }


def _validate_trace_outcome_model(*, root: Path) -> dict:
    trace_dataset_candidates = [
        root / "phase four/assumption_graph/trace_dataset_collection_ms_bridge_20260601.json",
        root / "phase four/assumption_graph/trace_dataset_ms_bridge_20260601.json",
    ]
    trace_dataset_path = next((path for path in trace_dataset_candidates if path.exists()), trace_dataset_candidates[0])
    if not trace_dataset_path.exists():
        return {
            "pass": False,
            "reason": "missing_trace_dataset_artifact",
            "trace_dataset_path": _display_path(root, trace_dataset_path),
        }
    collection_mode = "collection" in trace_dataset_path.name
    payload = build_trace_outcome_model_payload(
        trace_dataset_payload=json.loads(trace_dataset_path.read_text(encoding="utf-8")),
        eval_id="perf_trace_outcome_model",
        min_policy_group_size=2,
    )
    metrics = payload["leave_one_out_metrics"]
    feature_metrics = payload["feature_leave_one_out_metrics"]
    brier = metrics.get("weighted_brier_score")
    if brier is None:
        brier = metrics.get("brier_score")
    feature_brier = feature_metrics.get("weighted_brier_score")
    if feature_brier is None:
        feature_brier = feature_metrics.get("brier_score")
    loss_updates = [
        row for row in payload["policy_updates"]
        if row["decision"] in {"keep_with_targeted_repair", "repair_before_scaling"}
    ]
    min_rows = 60 if collection_mode else 9
    min_weighted_rows = 35.0 if collection_mode else 9.0
    min_routes = 4 if collection_mode else 3
    min_updates = 3 if collection_mode else 1
    max_brier = 0.12 if collection_mode else 0.25
    return {
        "pass": (
            payload["trainable_row_count"] >= min_rows
            and payload["weighted_trainable_row_count"] >= min_weighted_rows
            and payload["route_group_count"] >= min_routes
            and payload["policy_update_count"] >= min_updates
            and len(loss_updates) >= 1
            and brier is not None
            and brier <= max_brier
            and feature_brier is not None
            and feature_brier <= brier
            and not payload["secret_leak_detected"]
        ),
        "trace_dataset_path": _display_path(root, trace_dataset_path),
        "collection_mode": collection_mode,
        "trainable_row_count": payload["trainable_row_count"],
        "weighted_trainable_row_count": payload["weighted_trainable_row_count"],
        "trace_source_counts": payload["trace_source_counts"],
        "trace_source_weighted_counts": payload["trace_source_weighted_counts"],
        "route_group_count": payload["route_group_count"],
        "component_group_count": payload["component_group_count"],
        "residual_group_count": payload["residual_group_count"],
        "policy_update_count": payload["policy_update_count"],
        "loss_policy_update_count": len(loss_updates),
        "leave_one_out_metrics": metrics,
        "feature_leave_one_out_metrics": feature_metrics,
        "feature_schema": payload["feature_schema"],
        "route_stats": payload["route_stats"],
        "policy_updates": payload["policy_updates"],
        "secret_leak_detected": payload["secret_leak_detected"],
    }


def _validate_trace_policy_proposals(*, root: Path, graph_dir: Path) -> dict:
    trace_outcome_candidates = [
        root / "phase four/assumption_graph/trace_outcome_model_collection_ms_bridge_20260601.json",
        root / "phase four/assumption_graph/trace_outcome_model_ms_bridge_20260601.json",
    ]
    trace_outcome_path = next((path for path in trace_outcome_candidates if path.exists()), trace_outcome_candidates[0])
    if not trace_outcome_path.exists():
        return {
            "pass": False,
            "reason": "missing_trace_outcome_model_artifact",
            "trace_outcome_path": _display_path(root, trace_outcome_path),
        }
    collection_mode = "collection" in trace_outcome_path.name
    payload = build_trace_policy_proposal_payload(
        store=JsonlGraphStore(graph_dir),
        trace_outcome_payload=json.loads(trace_outcome_path.read_text(encoding="utf-8")),
        eval_id="perf_trace_policy_proposals",
    )
    proposals = payload.get("proposals", [])
    repair_count = sum(
        1 for proposal in proposals
        if proposal.get("source_action", {}).get("decision") in {"keep_with_targeted_repair", "repair_before_scaling"}
    )
    candidate_count = sum(1 for proposal in proposals if proposal.get("candidate_node"))
    verifier_count = sum(
        1 for proposal in proposals
        if "heldout_route_ablation" in (proposal.get("candidate_node", {}).get("verifiers") or [])
    )
    min_proposals = 4 if collection_mode else 3
    return {
        "pass": (
            payload["parent_node_id"] is not None
            and payload["proposal_count"] >= min_proposals
            and candidate_count == payload["proposal_count"]
            and repair_count >= 1
            and verifier_count == payload["proposal_count"]
            and not payload["secret_leak_detected"]
        ),
        "trace_outcome_path": _display_path(root, trace_outcome_path),
        "collection_mode": collection_mode,
        "parent_node_id": payload["parent_node_id"],
        "proposal_count": payload["proposal_count"],
        "proposal_counts": payload["proposal_counts"],
        "decision_counts": payload["decision_counts"],
        "repair_policy_count": repair_count,
        "candidate_count": candidate_count,
        "heldout_verifier_count": verifier_count,
        "secret_leak_detected": payload["secret_leak_detected"],
    }


def _validate_trace_policy_preflight(*, root: Path) -> dict:
    preflight_candidates = [
        root / "phase four/assumption_graph/trace_policy_preflight_collection_ms_bridge_20260601.json",
        root / "phase four/assumption_graph/trace_policy_preflight_ms_bridge_20260601.json",
    ]
    preflight_path = next((path for path in preflight_candidates if path.exists()), preflight_candidates[0])
    if not preflight_path.exists():
        return {
            "pass": False,
            "reason": "missing_trace_policy_preflight_artifact",
            "preflight_path": _display_path(root, preflight_path),
        }
    collection_mode = "collection" in preflight_path.name
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    summaries = payload.get("summaries", [])
    ready = [row for row in summaries if row.get("readiness") == "ready_for_fresh_ablation"]
    missed = sum(len(row.get("missed_trigger_problem_ids", [])) for row in summaries)
    outside = sum(len(row.get("outside_active_problem_ids", [])) for row in summaries)
    command_hints = sum(1 for row in summaries if row.get("command_hint"))
    min_proposals = 4 if collection_mode else 3
    return {
        "pass": (
            len(summaries) >= min_proposals
            and len(ready) == len(summaries)
            and missed == 0
            and outside == 0
            and command_hints == len(summaries)
        ),
        "preflight_path": _display_path(root, preflight_path),
        "collection_mode": collection_mode,
        "proposal_count": len(summaries),
        "readiness_counts": payload.get("readiness_counts", {}),
        "ready_count": len(ready),
        "missed_trigger_count": missed,
        "outside_active_count": outside,
        "command_hint_count": command_hints,
    }


def _validate_harness_observer(*, root: Path, graph_dir: Path) -> dict:
    artifact_paths = [
        root / "phase two/analysis/cache/judgments/phase2_v20_gpt55_vs_phase2_v20_ms_bridge_gpt55_21_50.json",
        root / "phase two/analysis/cache/answers/phase2_v20_ms_bridge_gpt55_21_50_meta.json",
        root / "phase four/assumption_graph/recursive_scoped_judge_run_gpt55_21_50.log",
        root / "phase six/autonomous/exp80_run.log",
    ]
    existing = [path for path in artifact_paths if path.exists()]
    with tempfile.TemporaryDirectory() as td:
        temp_graph = Path(td) / "graph"
        _copy_graph_store(graph_dir, temp_graph)
        payload = build_harness_observer_payload(
            root=root,
            graph_dir=temp_graph,
            eval_id="perf_harness_observer",
            artifact_paths=existing,
            max_events_per_file=12,
            writeback=True,
        )
    coverage = payload["artifact_coverage"]
    leak = bool(payload.get("secret_leak_detected"))
    passed = (
        len(existing) >= 3
        and payload["discovered_event_count"] >= 10
        and payload["backfilled_event_count"] + payload["skipped_covered_event_count"] == payload["discovered_event_count"]
        and coverage["full_coverage_after_writeback"]
        and coverage["post_covered_file_count"] == coverage["event_artifact_file_count"]
        and not leak
    )
    return {
        "pass": passed,
        "artifact_file_count": coverage["artifact_file_count"],
        "discovered_event_count": payload["discovered_event_count"],
        "backfilled_event_count": payload["backfilled_event_count"],
        "skipped_covered_event_count": payload["skipped_covered_event_count"],
        "event_counts": payload["event_counts"],
        "discovered_event_counts": payload["discovered_event_counts"],
        "artifact_kind_counts": payload["artifact_kind_counts"],
        "post_covered_file_count": coverage["post_covered_file_count"],
        "uncovered_after_writeback": coverage["uncovered_after_writeback"],
        "full_coverage_after_writeback": coverage["full_coverage_after_writeback"],
        "secret_leak_detected": leak,
        "artifact_paths": coverage["artifact_paths"],
    }


def _validate_residual_clusterer(*, graph_dir: Path) -> dict:
    payload = build_residual_cluster_payload(
        store=JsonlGraphStore(graph_dir),
        eval_id="perf_residual_clusterer",
        min_cluster_size=2,
        writeback_manifests=False,
    )
    proposals = payload.get("proposals", [])
    validation_complete = all(
        p.get("candidate_node", {}).get("payload", {}).get("validation_plan", {}).get("trigger_problem_ids")
        for p in proposals
    )
    return {
        "pass": payload["record_count"] >= 20 and payload["cluster_count"] >= 2 and payload["proposal_count"] >= 2 and validation_complete,
        "record_count": payload["record_count"],
        "cluster_count": payload["cluster_count"],
        "proposal_count": payload["proposal_count"],
        "residual_type_counts": payload["residual_type_counts"],
        "proposal_parent_ids": [p["parent_node_id"] for p in proposals],
        "validation_plans_complete": validation_complete,
    }


def _validate_formal_metrics(*, root: Path, graph_dir: Path) -> dict:
    store = JsonlGraphStore(graph_dir)
    formal_payload = build_formal_mapping_payload(store)
    metric_payload = build_categorical_info_geometry_payload(formal_payload)
    dedup_payload = build_formal_dedup_payload(formal_payload)
    dedup_control = _formal_dedup_positive_control()
    search_eval_candidates = [
        root / "phase four/assumption_graph/formal_mapping_search_eval_expanded_phase2_graph.json",
        root / "phase four/assumption_graph/formal_mapping_search_eval_phase2_graph.json",
    ]
    search_eval_path = next((path for path in search_eval_candidates if path.exists()), None)
    search_eval_payload = (
        json.loads(search_eval_path.read_text(encoding="utf-8"))
        if search_eval_path
        else build_formal_search_eval_payload(formal_payload)
    )
    transfer_payload = build_formal_transfer_eval_payload(
        formal_mapping_payload=formal_payload,
        metric_payload=metric_payload,
        search_eval_payload=search_eval_payload,
    )
    summaries = metric_payload["summaries"]
    same_shape = sum(1 for row in summaries if row["metrics"].get("same_shape"))
    warning_count = sum(len(row.get("warnings", [])) for row in summaries)
    complete_count = formal_payload.get("status_counts", {}).get("complete", 0)
    dedup_control_ok = (
        dedup_control["duplicate_cluster_count"] == 1
        and dedup_control["merge_recommendation_count"] == 1
        and dedup_control["incomplete_mapping_excluded_count"] == 1
    )
    return {
        "pass": (
            complete_count >= 5
            and same_shape == len(summaries)
            and warning_count == 0
            and dedup_control_ok
            and transfer_payload.get("pass", False)
            and search_eval_payload.get("query_count", 0) >= complete_count
            and search_eval_payload.get("negative_application_count", 0) >= complete_count * max(0, complete_count - 1)
        ),
        "mapping_count": metric_payload["mapping_count"],
        "complete_count": complete_count,
        "same_shape_count": same_shape,
        "warning_count": warning_count,
        "metric_summary": metric_payload["metric_summary"],
        "dedup_pass": dedup_control_ok,
        "dedup_complete_mapping_count": dedup_payload["complete_mapping_count"],
        "dedup_unique_signature_count": dedup_payload["unique_signature_count"],
        "dedup_duplicate_cluster_count": dedup_payload["duplicate_cluster_count"],
        "dedup_merge_recommendation_count": dedup_payload["merge_recommendation_count"],
        "dedup_incomplete_mapping_excluded_count": dedup_payload["incomplete_mapping_excluded_count"],
        "dedup_positive_control": {
            "duplicate_cluster_count": dedup_control["duplicate_cluster_count"],
            "merge_recommendation_count": dedup_control["merge_recommendation_count"],
            "incomplete_mapping_excluded_count": dedup_control["incomplete_mapping_excluded_count"],
        },
        "transfer_eval_pass": transfer_payload.get("pass", False),
        "transfer_search_eval_path": _display_path(root, search_eval_path) if search_eval_path else "generated_in_memory",
        "transfer_search_query_count": search_eval_payload.get("query_count", 0),
        "transfer_search_top1_hit_rate": search_eval_payload.get("top1_hit_rate"),
        "transfer_search_negative_application_count": search_eval_payload.get("negative_application_count", 0),
        "transfer_query_count": transfer_payload.get("query_count", 0),
        "transfer_application_count": transfer_payload.get("application_count", 0),
        "transfer_top1_hit_rate": transfer_payload.get("top1_hit_rate"),
        "transfer_pairwise_auc": transfer_payload.get("pairwise_auc"),
        "transfer_positive_mean_score": transfer_payload.get("positive_mean_transfer_score"),
        "transfer_negative_mean_score": transfer_payload.get("negative_mean_transfer_score"),
    }


def _validate_surface_hypothesis_generator(*, graph_dir: Path, sections: dict[str, dict]) -> dict:
    payload = build_surface_hypothesis_payload(
        store=JsonlGraphStore(graph_dir),
        performance_sections=sections,
        eval_id="perf_surface_hypotheses",
    )
    proposals = payload.get("proposals", [])
    candidate_count = sum(1 for proposal in proposals if proposal.get("candidate_node"))
    manifest_count = payload.get("manifest_count", 0)
    verifier_count = sum(
        1 for proposal in proposals
        if (proposal.get("candidate_node", {}).get("verifiers") or [])
    )
    return {
        "pass": (
            payload["proposal_count"] >= 4
            and payload["world_model_proposal_count"] >= 2
            and payload["evaluator_proposal_count"] >= 2
            and candidate_count == payload["proposal_count"]
            and manifest_count == payload["proposal_count"]
            and verifier_count == payload["proposal_count"]
            and not payload["secret_leak_detected"]
        ),
        "proposal_count": payload["proposal_count"],
        "proposal_counts": payload["proposal_counts"],
        "surface_counts": payload["surface_counts"],
        "world_model_proposal_count": payload["world_model_proposal_count"],
        "evaluator_proposal_count": payload["evaluator_proposal_count"],
        "candidate_count": candidate_count,
        "manifest_count": manifest_count,
        "verifier_count": verifier_count,
        "secret_leak_detected": payload["secret_leak_detected"],
        "proposals": proposals,
    }


def _formal_dedup_positive_control() -> dict:
    with tempfile.TemporaryDirectory() as td:
        store = JsonlGraphStore(Path(td))
        for seed in ["WCAND_DUP_A", "WCAND_DUP_B"]:
            base = {
                "type": AssumptionType.HARNESS,
                "claim": f"formal dedup positive control {seed}",
                "payload": {"seed_cid": seed},
                "tags": [seed],
            }
            for suffix, kind, expr in [
                ("feature", "feature", {"keywords_en": ["risk"], "regex": []}),
                ("constraint", "constraint", {"required_substrings": ["rollback"]}),
                ("decomp", "decomposition", {"steps": ["identify", "verify"]}),
                ("verify", "verification", {"instruction": "check rollback"}),
                ("hp", "hp_change", {"temperature": 0.0, "max_tokens": 1000}),
            ]:
                store.upsert_node(AssumptionNode(
                    id=f"{seed}_{suffix}",
                    kind=kind,
                    formal_form={"kind": kind, "expr": expr},
                    **base,
                ))
        store.upsert_node(AssumptionNode(
            id="WCAND_UNSAFE_constraint",
            type=AssumptionType.HARNESS,
            kind="constraint",
            claim="unsafe duplicate should be excluded",
            formal_form={"kind": "constraint", "expr": {"required_substrings": ["rollback"]}},
            payload={"seed_cid": "WCAND_UNSAFE"},
            tags=["WCAND_UNSAFE"],
        ))
        formal_payload = build_formal_mapping_payload(store)
        return build_formal_dedup_payload(formal_payload)


def _combined_candidate_bundle(root: Path) -> dict:
    artifact_dir = root / DEFAULT_ARTIFACT_DIR
    cycle = _load_json(artifact_dir / "evolution_cycle_dryrun_phase2_v20_gpt55_21_50.json")
    proposal_screen = _load_json(artifact_dir / "proposals_phase2_v20_gpt55_21_50.json")
    proposal_preflight = _load_json(artifact_dir / "candidate_preflight_phase2_v20_gpt55_21_50.json")
    pos_ms = _load_json(artifact_dir / "recursive_positive_ms_bridge_evolution.json")
    pos_se = _load_json(artifact_dir / "recursive_positive_se_hard_policy_evolution.json")
    proposal_payload = _merge_proposal_payloads([
        cycle["proposals"],
        proposal_screen,
        pos_ms["proposals"],
        pos_se["proposals"],
    ])
    preflight_payload = _merge_list_payload("summaries", [
        cycle["candidate_preflight"],
        proposal_preflight,
        pos_ms["candidate_preflight"],
        pos_se["candidate_preflight"],
    ])
    formal_gate_payload = _merge_list_payload("gates", [
        cycle.get("formal_mapping_gate", {}),
        pos_ms.get("formal_mapping_gate", {}),
        pos_se.get("formal_mapping_gate", {}),
    ])
    regression_predictions = [
        *cycle.get("regression_predictions", []),
        *pos_ms.get("regression_predictions", []),
        *pos_se.get("regression_predictions", []),
    ]
    acceptance_sources = [
        _load_json(artifact_dir / "recursive_candidate_acceptance_phase2_v20_gpt55_21_50.json"),
        _load_json(artifact_dir / "candidate_acceptance_summary_gpt54mini_21_50.json"),
        _load_json(artifact_dir / "recursive_positive_ms_bridge_acceptance.json"),
        _load_json(artifact_dir / "recursive_positive_se_hard_policy_acceptance.json"),
    ]
    summaries = []
    for payload in acceptance_sources:
        summaries.extend(payload.get("summaries", []))
    labels = {}
    for row in summaries:
        decision = row.get("decision")
        if decision == "accept":
            labels[row["proposal_id"]] = "accept"
        elif decision in {"reject_benefit", "reject_harm"}:
            labels[row["proposal_id"]] = "reject"
    return {
        "proposal_payload": proposal_payload,
        "preflight_payload": preflight_payload,
        "falsification_payload": build_falsification_payload(
            proposal_payload=proposal_payload,
            preflight_payload=preflight_payload,
            acceptance_payload={
                "eval_id": "perf_combined_acceptance",
                "summaries": summaries,
                "accepted_proposal_ids": sorted(pid for pid, label in labels.items() if label == "accept"),
                "decision_counts": dict(Counter(row.get("decision") for row in summaries)),
            },
        ),
        "formal_gate_payload": formal_gate_payload,
        "regression_predictions": regression_predictions,
        "acceptance_payload": {
            "eval_id": "perf_combined_acceptance",
            "summaries": summaries,
            "accepted_proposal_ids": sorted(pid for pid, label in labels.items() if label == "accept"),
            "decision_counts": dict(Counter(row.get("decision") for row in summaries)),
        },
        "labels": labels,
    }


def _rank_metrics(predictions: list[dict], labels: dict[str, str]) -> dict:
    rows = [
        (p["proposal_id"], float(p["predicted_acceptance_probability"]), labels[p["proposal_id"]])
        for p in predictions
        if p["proposal_id"] in labels
    ]
    accepted = [score for _, score, label in rows if label == "accept"]
    rejected = [score for _, score, label in rows if label == "reject"]
    auc = _pairwise_auc(accepted, rejected)
    ranked = sorted(rows, key=lambda row: row[1], reverse=True)
    k = max(1, len(accepted))
    accepted_in_top_k = sum(1 for _, _, label in ranked[:k] if label == "accept")
    return {
        "labeled_count": len(rows),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_mean_probability": round(sum(accepted) / len(accepted), 4) if accepted else None,
        "rejected_mean_probability": round(sum(rejected) / len(rejected), 4) if rejected else None,
        "accepted_rejected_margin": (
            round((sum(accepted) / len(accepted)) - (sum(rejected) / len(rejected)), 4)
            if accepted and rejected
            else None
        ),
        "auc": auc,
        "accepted_recall_at_k": round(accepted_in_top_k / len(accepted), 4) if accepted else 0.0,
        "top_ranked": [
            {"proposal_id": pid, "probability": round(score, 4), "label": label}
            for pid, score, label in ranked[:5]
        ],
    }


def _pairwise_auc(accepted: list[float], rejected: list[float]) -> float | None:
    if not accepted or not rejected:
        return None
    wins = 0.0
    total = 0
    for a in accepted:
        for r in rejected:
            if a > r:
                wins += 1.0
            elif a == r:
                wins += 0.5
            total += 1
    return round(wins / total, 4)


def _merge_proposal_payloads(payloads: list[dict]) -> dict:
    proposals = []
    seen = set()
    for payload in payloads:
        for proposal in payload.get("proposals", []):
            pid = proposal.get("proposal_id")
            if pid in seen:
                continue
            seen.add(pid)
            proposals.append(proposal)
    return {
        "eval_id": "perf_combined_proposals",
        "proposal_counts": dict(Counter(p.get("proposal_type", "") for p in proposals)),
        "proposals": proposals,
    }


def _merge_list_payload(key: str, payloads: list[dict]) -> dict:
    rows = []
    seen = set()
    for payload in payloads:
        for row in payload.get(key, []):
            row_key = (row.get("proposal_id"), json.dumps(row, ensure_ascii=False, sort_keys=True))
            if row_key in seen:
                continue
            seen.add(row_key)
            rows.append(row)
    return {"eval_id": f"perf_combined_{key}", key: rows}


def _judgment_sets_from_bundle(root: Path, bundle: dict) -> list[JudgmentSet]:
    sets = []
    for run in bundle.get("runs", []):
        sets.append(JudgmentSet(
            candidate_variant=run["candidate_variant"],
            baseline_variant=run.get("baseline_variant") or bundle.get("baseline_variant"),
            judgment_paths=[_resolve(root, p) for p in run.get("judgments", [])],
            proposal_ids=run.get("proposal_ids", []),
        ))
    return sets


def _copy_graph_store(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("nodes.jsonl", "edges.jsonl", "evidence.jsonl", "trials.jsonl"):
        source = src / name
        if source.exists():
            shutil.copy2(source, dst / name)


def _strip_payload(world: dict) -> dict:
    return {k: v for k, v in world.items() if k != "post_acceptance_payload"}


def _key_metric(name: str, section: dict) -> str:
    if name == "world_model":
        return f"labels={section['matched_label_count']}, pre_auc={section['pre_acceptance']['auc']}, brier={section['post_calibration']['brier_score']}"
    if name == "trajectory_search":
        return f"multi_path={section['multi_path_rate']}, hit={section['top_path_label_hit_rate']}"
    if name == "metaproductivity_benchmark":
        return (
            f"queries={section.get('query_count', 0)}, "
            f"acp_meta={section.get('mean_acp_top_clade_metaproductivity')}, "
            f"imm_meta={section.get('mean_immediate_top_clade_metaproductivity')}"
        )
    if name == "verifier_stack":
        return (
            f"accepted={section['accepted_count']}, rejected={section['rejected_count']}, "
            f"protocols={section['falsification_protocol_candidate_count']}/{section['proposal_count']}"
        )
    if name == "recursive_daemon":
        return f"applied={section['accepted_apply_count']}/{section['case_count']}"
    if name == "recursive_audit":
        return f"score={section['min_closure_score']}, issues={section['critical_issue_count']}/{section['warning_issue_count']}"
    if name == "evolution_context":
        return f"decision={section['dry_policy_decision']}->{section['apply_policy_decision']}, resp={section['responsibility_status_counts']}"
    if name == "assumption_bench":
        return f"score={section['overall_score']}, passed={section['passed_capability_count']}/{section['capability_count']}"
    if name == "reconstruction_progress":
        return f"structure={section['structure_percent']}%, behavior={section['behavior_percent']}%, weighted={section['weighted_percent']}%"
    if name == "memory_surfaces":
        return f"types={section['before_node_type_count']}->{section['after_node_type_count']}, edges={section['before_edge_type_count']}->{section['after_edge_type_count']}"
    if name == "manifest_logger":
        return f"events={section['event_count']}, real_logs={section['real_log_event_count']}, leak={section['secret_leak_detected']}"
    if name == "runtime_trace":
        return f"events={section['event_count']}, written={section['written_trials']}, leak={section['secret_leak_detected']}"
    if name == "trace_dataset":
        return f"rows={section['trainable_row_count']}/{section['row_count']}, coverage={section['traced_outcome_coverage']}, leak={section['secret_leak_detected']}"
    if name == "trace_outcome_model":
        metrics = section.get("leave_one_out_metrics", {})
        return f"rows={section.get('trainable_row_count', 0)}, brier={metrics.get('brier_score')}, updates={section.get('policy_update_count', 0)}"
    if name == "trace_policy_proposals":
        return f"proposals={section.get('proposal_count', 0)}, repair={section.get('repair_policy_count', 0)}, parent={section.get('parent_node_id')}"
    if name == "trace_policy_preflight":
        return f"ready={section.get('ready_count', 0)}/{section.get('proposal_count', 0)}, missed={section.get('missed_trigger_count', 0)}, outside={section.get('outside_active_count', 0)}"
    if name == "harness_observer":
        return f"artifacts={section['artifact_file_count']}, backfill={section['backfilled_event_count']}/{section['discovered_event_count']}, covered={section['full_coverage_after_writeback']}"
    if name == "residual_clusterer":
        return f"clusters={section['cluster_count']}, proposals={section['proposal_count']}"
    if name == "surface_hypothesis_generator":
        return (
            f"proposals={section.get('proposal_count', 0)}, "
            f"world={section.get('world_model_proposal_count', 0)}, "
            f"evaluator={section.get('evaluator_proposal_count', 0)}"
        )
    if name == "formal_metrics":
        return (
            f"mappings={section['mapping_count']}, warnings={section['warning_count']}, "
            f"dedup={section.get('dedup_duplicate_cluster_count', 0)}, "
            f"transfer_auc={section.get('transfer_pairwise_auc')}"
        )
    return ""


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 4)


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--eval-id", default="reconstruction_gap_performance_validation")
    ap.add_argument("--summary-out", default=None)
    ap.add_argument("--report-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    graph_dir = _resolve(root, args.graph_dir)
    payload = build_performance_validation_payload(
        root=root,
        graph_dir=graph_dir,
        eval_id=args.eval_id,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    if args.report_out:
        out = _resolve(root, args.report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(format_performance_report(payload), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
