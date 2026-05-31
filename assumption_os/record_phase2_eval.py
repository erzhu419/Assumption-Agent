"""Write Phase2 pairwise evaluation outcomes back into the Assumption Graph.

The evaluator cache says whether an Assumption-Graph-augmented variant beat a
baseline variant.  This module turns each judged problem into a TrialManifest:
which assumptions were active, what they predicted, what the judge observed,
and whether any residual should be attached to the active assumptions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from .schema import (
    AssumptionType,
    EvidenceRecord,
    ResidualType,
    TrialManifest,
    TrialStatus,
    stable_id,
)


PRIMARY_TYPES = {
    AssumptionType.METHOD,
    AssumptionType.HARNESS,
    AssumptionType.RETRIEVAL,
    AssumptionType.WORLD_MODEL,
    AssumptionType.EVALUATOR,
    AssumptionType.ALIGNMENT,
}


def record_phase2_eval(
    *,
    root: Path,
    graph_dir: Path,
    sample_path: Path,
    meta_path: Path,
    judgment_paths: Iterable[Path],
    intervention_variant: str,
    baseline_variant: str,
    eval_id: str,
    top_k: int = 8,
    policy_rerank: bool = False,
    skip_domains: set[str] | None = None,
    skip_missing_meta: bool = True,
    dry_run: bool = False,
) -> dict:
    judgment_paths = list(judgment_paths)
    store = JsonlGraphStore(graph_dir)
    graph = SimpleAssumptionGraph(store)
    sample = _load_json(sample_path)
    meta_by_pid = _load_json(meta_path)
    problems = {p["problem_id"]: p for p in sample if "problem_id" in p}
    before_conf = {nid: node.confidence for nid, node in store.nodes.items()}

    outcome_counts: Counter[str] = Counter()
    residual_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    processed = []
    planned_trials: list[tuple[TrialManifest, dict]] = []

    for judgment_path in judgment_paths:
        judgments = _load_json(judgment_path)
        for pid, judgment in judgments.items():
            problem = problems.get(pid)
            if not problem:
                skipped["missing_problem"] += 1
                continue
            meta = meta_by_pid.get(pid)
            if skip_missing_meta and not meta:
                skipped["missing_meta"] += 1
                continue

            subgraph = retrieve_eval_subgraph(
                graph,
                problem,
                meta or {},
                top_k=top_k,
                policy_rerank=policy_rerank,
                skip_domains=skip_domains,
            )
            if subgraph is None:
                skipped["policy_skipped"] += 1
                continue
            trial, info = build_trial_manifest(
                eval_id=eval_id,
                problem=problem,
                meta=meta or {},
                judgment=judgment,
                judgment_path=judgment_path,
                intervention_variant=intervention_variant,
                baseline_variant=baseline_variant,
                active_assumption_ids=[node.id for node in subgraph.nodes],
            )

            outcome_counts[info["outcome"]] += 1
            residual_counts[str(trial.residual_type.value if hasattr(trial.residual_type, "value") else trial.residual_type)] += 1
            processed.append(info)
            planned_trials.append((trial, judgment))

    if not dry_run:
        for trial, judgment in planned_trials:
            graph.update_from_trial(trial, persist=False)
            evidence_value = {"win": 1.0, "loss": 0.0, "tie": 0.5}.get(trial.metadata.get("outcome"))
            evidence_outcome = {
                "win": "success",
                "loss": "failed",
                "tie": "tie",
            }.get(trial.metadata.get("outcome"), "observed")
            for node_id in trial.assumption_ids:
                store.add_evidence(EvidenceRecord(
                    node_id=node_id,
                    source=eval_id,
                    outcome=evidence_outcome,
                    metric="pairwise_judge_win",
                    value=evidence_value,
                    split="phase2_sample20",
                    details={
                        "trial_id": trial.trial_id,
                        "problem_id": trial.problem_id,
                        "judgment_path": trial.artifacts.get("judgment_path"),
                        "winner": judgment.get("winner"),
                        "reasoning": judgment.get("reasoning", ""),
                        "score_a": judgment.get("score_a"),
                        "score_b": judgment.get("score_b"),
                    },
                    evidence_id=stable_id("ev", trial.trial_id, node_id),
                ))

    if not dry_run:
        store.flush()
        graph.reindex()

    after_conf = {nid: node.confidence for nid, node in store.nodes.items()}
    changed = {
        nid: {
            "before": before_conf.get(nid),
            "after": after_conf[nid],
            "delta": after_conf[nid] - before_conf.get(nid, after_conf[nid]),
        }
        for nid in sorted(after_conf)
        if abs(after_conf[nid] - before_conf.get(nid, after_conf[nid])) > 1e-9
    }

    by_domain = defaultdict(Counter)
    for item in processed:
        by_domain[item["domain"]][item["outcome"]] += 1

    return {
        "eval_id": eval_id,
        "dry_run": dry_run,
        "graph_dir": str(graph_dir),
        "sample_path": str(sample_path),
        "meta_path": str(meta_path),
        "judgment_paths": [str(p) for p in judgment_paths],
        "intervention_variant": intervention_variant,
        "baseline_variant": baseline_variant,
        "top_k": top_k,
        "policy_rerank": policy_rerank,
        "skip_domains": sorted(skip_domains or []),
        "processed": len(processed),
        "skipped": dict(skipped),
        "outcomes": dict(outcome_counts),
        "residual_types": dict(residual_counts),
        "by_domain": {dom: dict(counts) for dom, counts in sorted(by_domain.items())},
        "confidence_changes": changed,
        "processed_trials": processed,
    }


def retrieve_eval_subgraph(
    graph: SimpleAssumptionGraph,
    problem: dict,
    meta: dict,
    *,
    top_k: int,
    policy_rerank: bool = False,
    skip_domains: set[str] | None = None,
):
    if policy_rerank:
        from .retrieval_policy import retrieve_phase2_assumptions

        result = retrieve_phase2_assumptions(
            graph,
            problem=problem.get("description", ""),
            meta=meta,
            pid=problem.get("problem_id", ""),
            domain=problem.get("domain", ""),
            difficulty=problem.get("difficulty", ""),
            top_k=top_k,
            pool_k=max(24, top_k),
            skip_domains=skip_domains,
        )
        return result.subgraph if result else None

    query = "\n".join([
        problem.get("description", ""),
        meta.get("critical_reframe", ""),
        meta.get("rewritten_problem", ""),
        meta.get("what_changed", ""),
    ])
    seeds = [
        problem.get("problem_id", ""),
        problem.get("domain", ""),
        problem.get("difficulty", ""),
        meta.get("frame", ""),
        *meta.get("anti_patterns", [])[:3],
    ]
    return graph.retrieve(query, seeds=seeds, top_k=top_k, candidate_types=PRIMARY_TYPES)


def build_trial_manifest(
    *,
    eval_id: str,
    problem: dict,
    meta: dict,
    judgment: dict,
    judgment_path: Path,
    intervention_variant: str,
    baseline_variant: str,
    active_assumption_ids: list[str],
) -> tuple[TrialManifest, dict]:
    pid = problem["problem_id"]
    winner = judgment.get("winner", "tie")
    gold_ids = {f"strategy_{sid}" for sid in problem.get("coverage_tags", [])}
    gold_hit = bool(gold_ids & set(active_assumption_ids))

    if winner == intervention_variant:
        outcome = "win"
        status = TrialStatus.ACCEPTED
        residual_type = ResidualType.NO_RESIDUAL
        residual = None
        observed = "Pairwise judge preferred the Assumption Graph variant."
    elif winner == baseline_variant:
        outcome = "loss"
        status = TrialStatus.FAILED
        residual_type = ResidualType.OPTIMIZATION if gold_hit else ResidualType.MEMORY_DEFECT
        observed = "Pairwise judge preferred the same-model baseline over the Assumption Graph variant."
        residual = _loss_residual(judgment, gold_hit=gold_hit)
    else:
        outcome = "tie"
        status = TrialStatus.DEFERRED
        residual_type = ResidualType.UNKNOWN
        residual = None
        observed = "Pairwise judge returned tie or undecided."

    trial = TrialManifest(
        problem_id=pid,
        action_type="phase2_assumption_graph_context",
        component="phase2_v20_turn1_context",
        assumption="Injecting the activated primary Assumption Graph subgraph should improve the final v20 answer.",
        why_selected="The v20 Turn0 frame/rewrite and problem text activated these primary graph assumptions.",
        expected_effect=f"{intervention_variant} should be preferred over {baseline_variant} by pairwise judge.",
        assumption_ids=active_assumption_ids,
        verifier="pairwise_gpt55_judge",
        verification_plan="Compare intervention and same-model baseline with randomized A/B side assignment.",
        rollback_condition="If baseline wins systematically, inspect retrieval hits and residual types before scaling.",
        observed_effect=observed,
        residual=residual,
        residual_type=residual_type,
        status=status,
        artifacts={
            "judgment_path": str(judgment_path),
            "intervention_variant": intervention_variant,
            "baseline_variant": baseline_variant,
        },
        metadata={
            "eval_id": eval_id,
            "winner": winner,
            "outcome": outcome,
            "reasoning": judgment.get("reasoning", ""),
            "score_a": judgment.get("score_a"),
            "score_b": judgment.get("score_b"),
            "a_was": judgment.get("a_was"),
            "domain": problem.get("domain"),
            "difficulty": problem.get("difficulty"),
            "frame": meta.get("frame"),
            "gold_ids": sorted(gold_ids),
            "gold_hit": gold_hit,
            "active_assumption_ids": active_assumption_ids,
        },
        trial_id=stable_id("trial", eval_id, judgment_path.name, pid, intervention_variant, baseline_variant),
    )
    return trial, {
        "trial_id": trial.trial_id,
        "problem_id": pid,
        "domain": problem.get("domain"),
        "difficulty": problem.get("difficulty"),
        "outcome": outcome,
        "winner": winner,
        "residual_type": residual_type.value,
        "gold_hit": gold_hit,
        "gold_ids": sorted(gold_ids),
        "active_assumption_ids": active_assumption_ids,
        "judgment_path": str(judgment_path),
    }


def _loss_residual(judgment: dict, *, gold_hit: bool) -> str:
    reason = judgment.get("reasoning", "")
    if gold_hit:
        prefix = "Graph retrieved at least one gold strategy, but the answer did not convert that context into a judge-preferred response."
    else:
        prefix = "Graph retrieval missed the sample's gold strategy tags for this problem."
    return f"{prefix} Judge reasoning: {reason}".strip()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="project root")
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--sample", default="phase two/analysis/cache/sample_100.json")
    ap.add_argument("--meta", required=True, help="intervention meta JSON")
    ap.add_argument("--judgments", nargs="+", required=True, help="one or more judgment JSON files")
    ap.add_argument("--intervention", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--policy-rerank", action="store_true",
                    help="attribute trials using the same domain-aware retrieval policy as Phase2 graph injection")
    ap.add_argument("--assumption-graph-skip-domains", default="",
                    help="comma-separated domains to skip when --policy-rerank is active")
    ap.add_argument("--include-missing-meta", action="store_true",
                    help="record trials even for v20 math/science bypass rows without graph meta")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    summary = record_phase2_eval(
        root=root,
        graph_dir=_resolve(root, args.graph_dir),
        sample_path=_resolve(root, args.sample),
        meta_path=_resolve(root, args.meta),
        judgment_paths=[_resolve(root, p) for p in args.judgments],
        intervention_variant=args.intervention,
        baseline_variant=args.baseline,
        eval_id=args.eval_id,
        top_k=args.top_k,
        policy_rerank=args.policy_rerank,
        skip_domains={x.strip() for x in args.assumption_graph_skip_domains.split(",") if x.strip()},
        skip_missing_meta=not args.include_missing_meta,
        dry_run=args.dry_run,
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
