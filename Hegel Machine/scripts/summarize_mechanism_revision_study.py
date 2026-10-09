#!/usr/bin/env python3
"""Read recorded mechanism proposals, experiments and final adoption; do no research."""

import argparse
from collections import Counter
import sys

from run_knowledge_lineage_study import ROOT, COSTS, read, save


def revision_summary(revision):
    if not revision:
        return None
    removed = revision.get("removed_term")
    return {"removed_operator": revision.get("removed_operator"),
            "removed_term_op": removed.get("op") if removed else None,
            **{key: revision.get(key) for key in
               ("source_ids", "status", "probe_id", "measured_loss")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    protocol = read(output / "protocol.json")
    sys.path.insert(0, str(ROOT / protocol["arms"]["knowledge_revisit"]["runtime"]))
    from hegel_machine.cognition.language import Program

    rows, totals = [], {}
    for arm in protocol["arms"]:
        counts = Counter()
        for root in protocol["roots"]:
            path = output / "runs" / arm / root
            state = read(path / "checkpoint.json.gz")
            terminal = read(path / "evaluation.json")
            for position, result in enumerate(state["results"]):
                report = result["research"]
                proposed, experiments = [], []
                for action in report["actions"]:
                    if action["action"] == "propose_mechanism_revisions":
                        counts["proposal_batches"] += 1
                        for item in action["alternatives"]:
                            proposed.append({"kind": item["kind"],
                                "candidate_id": item["candidate_id"],
                                "revision": revision_summary(item["revision"])})
                            counts["proposed_" + item["kind"]] += 1
                    elif action["action"] == "test_mechanism_predictions":
                        evidence = action["evidence"]
                        programs = {h["candidate_id"]: Program.from_dict(h["program"])
                            for h in action["plan"]["hypotheses"]}
                        hypotheses = [{**{key: h.get(key) for key in
                            ("candidate_id", "status", "measured_loss", "rows")},
                            "program_size": programs[h["candidate_id"]].expanded().size}
                            for h in evidence["hypotheses"]]
                        experiments.append({"probe_id": action["probe_id"],
                            "status": evidence["status"],
                            "observation_rows": evidence["observation_rows"],
                            "hypotheses": hypotheses})
                        counts["actual_experiments"] += 1
                        counts["experimental_rows"] += evidence["observation_rows"]
                        counts["original_refuted"] += sum(h["candidate_id"] == "mechanism_original"
                            and h["status"] == "counterexample" for h in hypotheses)
                revision = report.get("mechanism_revision")
                if revision:
                    counts["selected_" + report["selected_kind"]] += 1
                    counts["selected_" + revision["status"]] += 1
                    operator = revision.get("removed_operator")
                    revision_type = "deletion" if operator or revision.get("removed_term") else "full_refit"
                    counts["selected_revision_" + revision_type] += 1
                    if revision["status"] == "supported_on_fresh_trajectory" and report["status"] == "completed":
                        counts["confirmed_revision_" + revision_type] += 1
                    if operator:
                        counts["selected_removed_" + operator["op"]] += 1
                relation_experiments = sum(a["action"] == "test_inherited_relation_predictions"
                    for a in report["actions"])
                counts["retained_relation_experiments"] += relation_experiments
                relation_steps = [{"operation": step["operation"],
                    "parameter_count": len(step.get("parameters", [])),
                    "evidence_context_count": len(step.get("evidence_contexts", []))}
                    for step in (report.get("research_program") or {}).get("selection_steps", [])
                    if step["operation"] == "resolve_relation_intervals"]
                rows.append({"arm": arm, "root": root, "task_position": position,
                    "stage": protocol["stages"][position], "selected_kind": report["selected_kind"],
                    "status": report["status"], "mse": terminal[position]["mse"],
                    "final_program_size": Program.from_dict(report["program"]).expanded().size,
                    "final_channels": report["selected_channels"],
                    "retained_relation_steps": relation_steps,
                    "retained_relation_experiments": relation_experiments,
                    "selected_revision": revision_summary(revision), "proposed": proposed,
                    "experiments": experiments})
        totals[arm] = dict(counts)
    save(output / "mechanism_outcomes.json", {"arm_totals": totals, "tasks": rows,
        "size_scope": "Compare hypotheses within the same experiment. Final command-projected programs can be larger than measured-coordinate hypotheses and are not directly comparable."})
    summary = read(output / "summary.json")
    costs = {}
    for arm, total in summary["arm_totals"].items():
        costs[arm] = {key: total[key] for key in
            ("source_costs", "learning_costs", "terminal_costs", "source_cpu_seconds", "cpu_seconds")}
        costs[arm]["all_costs"] = {key: sum(total[part][key] for part in
            ("source_costs", "learning_costs", "terminal_costs")) for key in COSTS}
        costs[arm]["source_plus_research_cpu_seconds"] = total["source_cpu_seconds"] + total["cpu_seconds"]
        costs[arm]["final_deployment_measurement_cost"] = sum(
            row["deployment_measurement_costs"]["final"] for row in summary["results"] if row["arm"] == arm)
    save(output / "cost_totals.json", costs)
    print({"mechanism_outcomes": str(output / "mechanism_outcomes.json"), "arm_totals": totals})


if __name__ == "__main__":
    main()
