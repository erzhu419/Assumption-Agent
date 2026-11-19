#!/usr/bin/env python3
"""Narrow offline controls for V16.6 formation and intervention semantics.

Loads the fixed snapshot's real M1 modules; no numerical/function stubs, LLM,
network, repository writes, or full benchmark reproduction. Namespace shells
only avoid executing unrelated package __init__ files. Counterfactual inputs
and one explicitly named retrieval-policy subclass are local controls.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types


def load_modules(repo: Path):
    src = repo / "src" / "hegel_machine"
    for name, path in [("hegel_machine", src), ("hegel_machine.cognition", src / "cognition")]:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg
    loaded = {}
    for short in ["question_semantics", "wormhole_forest", "family_plugins", "cross_domain_clusters", "cross_domain_arms"]:
        name = "hegel_machine.cognition." + short
        path = src / "cognition" / (short + ".py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded[short] = module
    return loaded


def summarize_target(researcher, name, pair):
    factory, evaluator = pair
    task = factory()
    submission = researcher.solve_target(name, task)
    return {
        "submission": {k: v for k, v in submission.items() if k != "predict_fn"},
        "evaluation": evaluator(submission),
        "costs": task.meter.snapshot(),
        "observations": deepcopy(getattr(task, "observations", getattr(task, "query_log", []))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="Hegel Machine directory in the fixed source snapshot")
    parser.add_argument("--output", type=Path, default=Path("formation_controls.json"))
    args = parser.parse_args()
    mods = load_modules(args.repo.resolve())
    arms = mods["cross_domain_arms"]
    clusters = mods["cross_domain_clusters"]
    seed = 3026101
    ca = clusters.make_cluster_a_suite(seed)
    cb = clusters.make_cluster_b_suite(seed)
    targets = dict(cb["targets"])

    baseline = arms.HegelV166Researcher()
    baseline_logs = baseline.form_and_revise_from_experience("audit_cluster", ca, cb)
    results = {
        "scope": __doc__,
        "commit": "675f52612a9d0f1fb208da8bdb0552de3c571758",
        "seed": seed,
        "gate_scope": "Unmodified default gate, fresh identical deep-copied state for each target/control; not the study's development-calibrated total-score replay.",
        "source_sha256": {short: hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest() for short, mod in mods.items()},
    }

    # 1. Actual source/near-miss evidence retained in the shared archive.
    near_logs = [r for r in baseline_logs if r["phase"] == "source_nearmiss"]
    results["baseline_evidence_retention"] = {
        "source_and_nearmiss_episode_count": len(baseline_logs),
        "archive_episode_count": len(baseline.forest.raw_episode_archive),
        "archive_episode_ids": [r["episode_name"] for r in baseline.forest.raw_episode_archive],
        "near_miss_rows": [{"episode_name": r["episode_name"], "keys": sorted(r), "has_observations": "observations" in r or "query_log" in r} for r in near_logs],
        "forest_question_count": len(baseline.forest.questions),
        "node_question_ids": {nid: n["question_id"] for nid, n in baseline.forest.nodes.items()},
        "root_utility": baseline.forest.nodes["sys_root_concave_limit"]["utility_ledger"],
        "wormhole_endpoint_versions": [{"edge": e["wormhole_id"], "recorded_a": e["node_a_version"], "live_a": baseline.forest.nodes[e["node_a_id"]]["version"], "recorded_b": e["node_b_version"], "live_b": baseline.forest.nodes[e["node_b_id"]]["version"]} for e in baseline.forest.horizontal_wormhole_edges.values()],
    }

    # 2. Both ordered sources have no opposite endpoint and do not solve original task.
    ca_failed = deepcopy(ca)
    substituted = []
    for name, (factory, evaluator) in ca["sources"]:
        def no_flip_factory(original=factory):
            task = original()
            task._oracle = lambda _: 0
            return task
        substituted.append((name, (no_flip_factory, evaluator)))
    ca_failed["sources"] = substituted
    failed_former = arms.HegelV166Researcher()
    failed_logs = failed_former.form_and_revise_from_experience("audit_failed_sources", ca_failed, cb)
    a_ids = {name for name, _ in ca["sources"]}
    results["ordered_sources_no_support"] = {
        "control": "Only source oracles changed to constant zero; original evaluator remains, so original requested boundary is unsolved. Near-miss is unchanged.",
        "source_receipts": [{"episode_name": r["episode_name"], "strict_pass": r["strict_pass"], "unique_observed_labels": sorted(set(q[1] for q in r.get("query_log", [])))} for r in failed_logs if r["episode_name"] in a_ids],
        "ci_initial_claim": failed_former.forest.node_history["ci_mid_bisect"][0]["claim_contract"],
        "dac_claim": failed_former.forest.nodes["dac_mid_bisect"]["claim_contract"],
        "ordered_wormhole_created": "wh_A_ci_dac_bisect" in failed_former.forest.horizontal_wormhole_edges,
    }

    # 3. Duplicate the exact same source factory, evaluator and episode identity.
    cb_duplicate = deepcopy(cb)
    cb_duplicate["sources"] = [cb["sources"][0], cb["sources"][0]]
    duplicate = arms.HegelV166Researcher()
    duplicate.form_and_revise_from_experience("audit_duplicate_source", ca, cb_duplicate)
    dup_edge = duplicate.forest.horizontal_wormhole_edges.get("wh_B_sys_bio_rational")
    results["duplicate_source_creates_cross_domain_edge"] = {
        "control": "B source 2 replaced by exact source 1 tuple, including same episode ID and simulator.",
        "edge": dup_edge,
        "source_domains": [r["surface_domain"] for r in duplicate.forest.raw_episode_archive if r.get("induced_family")],
        "induced_families": [r["induced_family"] for r in duplicate.forest.raw_episode_archive if r.get("induced_family")],
        "sys_observation_map": duplicate.forest.nodes["sys_mid_rational"]["observation_map"],
        "bio_observation_map": duplicate.forest.nodes["bio_mid_rational"]["observation_map"],
    }

    # 4. A genuine joint response but a capped system where target confirmation fails.
    cb_capped = deepcopy(cb)
    nm_name, (nm_factory, _) = cb["near_misses"][0]
    capped_instances = []
    def capped_factory():
        task = nm_factory()
        task._sim = lambda ctrl: min(2.0 * ctrl["u1"], 2.0 * ctrl["u2"], 10.0)
        capped_instances.append(task)
        return task
    def capped_evaluator(sub):
        controls = sub.get("recommended_controls", {})
        achieved = min(2.0 * controls.get("u1", 0.0), 2.0 * controls.get("u2", 0.0), 10.0)
        ok = achieved >= 16.0 * 0.97
        return {"strict_pass": ok, "false_confirmation": not ok, "achieved_output": achieved, "required_output": 16.0}
    cb_capped["near_misses"][0] = (nm_name, (capped_factory, capped_evaluator))
    capped = arms.HegelV166Researcher()
    capped_logs = capped.form_and_revise_from_experience("audit_capped_confirmation", ca, cb_capped)
    results["failed_confirmation_still_promotes_bottleneck"] = {
        "control": "Near-miss B1 now has real min resource response capped at 10 while requested target remains 16; evaluator changed consistently with simulator.",
        "receipt": next(r for r in capped_logs if r["episode_name"] == nm_name),
        "actual_nearmiss_observations": deepcopy(capped_instances[-1].observations),
        "promoted_node_claim": capped.forest.nodes["bot_mid_limiting_factor_cegar"]["claim_contract"],
        "revision_event": next(r for r in capped.revision_events if r["witness_episode"] == nm_name),
    }

    # 5. Isolate explicit domain-local prohibition from the presence of graph edges.
    class CrossDomainLocalAllowed(arms.HegelV166Researcher):
        def _filter_cross_domain_candidates(self, recalled, target_domain):
            return recalled
    no_edge = deepcopy(baseline)
    no_edge.allow_horizontal_wormholes = False
    no_edge.allow_raw_history_rebuild_on_missing = False
    no_edge_local = deepcopy(no_edge)
    no_edge_local.__class__ = CrossDomainLocalAllowed
    results["edge_ablation_vs_domain_restriction"] = {
        "control": "Third condition changes only _filter_cross_domain_candidates to return existing local matches. It is a diagnostic policy control, not a complete equal-content Flat baseline or a proposed production fix.",
        "targets": {},
    }
    for target in ["T_B1_unseen_rational_transport", "T_B3_tied_bottleneck_joint"]:
        results["edge_ablation_vs_domain_restriction"]["targets"][target] = {
            "full": summarize_target(deepcopy(baseline), target, targets[target]),
            "no_horizontal_as_runner": summarize_target(deepcopy(no_edge), target, targets[target]),
            "no_horizontal_allow_existing_local_cross_domain": summarize_target(deepcopy(no_edge_local), target, targets[target]),
        }

    # 6. Does ordinary consume the same raw evidence at all?
    ordinary_empty = arms.StrongOrdinaryResearcher()
    ordinary_history = arms.StrongOrdinaryResearcher()
    ordinary_history.raw_archive = deepcopy(baseline.forest.raw_episode_archive)
    name = "T_B1_unseen_rational_transport"
    empty_result = summarize_target(ordinary_empty, name, targets[name])
    history_result = summarize_target(ordinary_history, name, targets[name])
    results["ordinary_history_consumption"] = {
        "empty": empty_result,
        "same_source_history": history_result,
        "exact_identical_observation_and_cost_trace": empty_result == history_result,
        "rebuilt_cache_after": ordinary_history.rebuilt_cache,
    }

    # 7. The learned near-miss exclusion versus the already hard-coded held-out gate.
    without_align_exclusion = deepcopy(baseline)
    without_align_exclusion.forest.horizontal_wormhole_edges["wh_B_sys_bio_rational"]["align_conditions"]["exclusion_predicates"] = []
    name = "T_B5_unbounded_sqrt_concave"
    original_trial = deepcopy(baseline)
    original_result = summarize_target(original_trial, name, targets[name])
    without_result = summarize_target(without_align_exclusion, name, targets[name])
    results["near_miss_exclusion_consumption"] = {
        "control": "Remove only learned exclusion predicate from the active rational wormhole; retain all source history, nodes, other metadata and the target algorithm.",
        "with_exclusion": original_result,
        "without_exclusion": without_result,
        "exact_identical_observation_and_cost_trace": original_result == without_result,
        "authorization_with": original_trial.alignment_checks[-1].get("authorization"),
        "authorization_without": without_align_exclusion.alignment_checks[-1].get("authorization"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "archive_vs_experiences": [results["baseline_evidence_retention"]["archive_episode_count"], len(baseline_logs)],
        "all_source_failures_still_edge": results["ordered_sources_no_support"]["ordered_wormhole_created"],
        "duplicate_source_still_edge": dup_edge is not None,
        "failed_confirmation_promoted": results["failed_confirmation_still_promotes_bottleneck"]["promoted_node_claim"]["evidence_status"],
        "ordinary_history_same_trace": results["ordinary_history_consumption"]["exact_identical_observation_and_cost_trace"],
        "remove_align_exclusion_same_trace": results["near_miss_exclusion_consumption"]["exact_identical_observation_and_cost_trace"],
        "target_controls": {t: {arm: {"pass": r["evaluation"]["strict_pass"], "meas": r["costs"]["measurement_cost"], "compute": r["costs"]["compute_cost"]} for arm, r in conditions.items()} for t, conditions in results["edge_ablation_vs_domain_restriction"]["targets"].items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
