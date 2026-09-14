#!/usr/bin/env python3
"""Remove active continuation planning from the frozen v0.48 development replay."""

import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import shutil

import run_sc_kg_objective_validation_study as validation


study = validation.study
ARM = "passive_continuation"
ABLATION = ("Keep the frozen v0.48 candidate retention, four-to-six-to-eight validation "
    "rounds and ordinary paid-label sharing. In the experimental source, return immediately "
    "from ObjectiveResearch._research_scope when an existing validating scope has at least "
    "four labels, before scanning or forecasting its active continuation pool. Also replace "
    "plan_validation_round with return None. Initial four-label validation, collecting, "
    "observation updates, checkpoints and shared-label validation remain unchanged.")


def comparison_references(source, spec):
    baseline = {row["case_id"]: row for row in
                study.read(spec["baseline_actual_source"])["results"]}
    references = study.read(source / "replay_summary.json")["result_references"]
    full = {row["case_id"]: study.read(row["case_record"]) for row in references
            if row["arm"] == validation.ARMS[1]}
    for key in spec["replay_cases"]:
        baseline[key], full[key]
    return baseline, full


def prepare(source, output):
    original = study.read(source / "protocol.json")
    by_id = {case["case_id"]: case for case in original["cases"]}
    cases = [deepcopy(by_id[key]) for key in original["replay_cases"]]
    frozen = output / "method_source_passive"
    spec = deepcopy(original)
    for field in ("affected_cases", "unchanged_entry_reuse", "actual_unchanged_controls",
                  "missing_actual_baseline_cases", "fresh_seeds", "fresh_cases", "reuse_rule"):
        spec.pop(field, None)
    spec.update(experiment="passive_objective_validation_continuation_ablation",
        cases=cases, arms=[ARM], source_validation_study=str(source.resolve()),
        frozen_method_source=str(frozen.resolve()),
        frozen_method_origin=original["frozen_method_source"],
        comparison_references=str((source / "replay_summary.json").resolve()),
        expected_new_optimizer_runs=len(cases), expected_new_online_calls=32 * len(cases),
        expected_new_initial_calls=0, expected_new_evaluation_calls=0,
        scope="The same sixteen prespecified v0.48 development replay cases; one new passive arm.",
        new_method=ABLATION,
        reuse_accounting="Reuse all already paid initial observations and evaluator measurements, "
            "and completed v0.47/v0.48 reference outcomes. Sixteen new passive optimizations cost "
            "512 online calls. No new initial/evaluator calls, fresh cases, or vanilla/CodeAgent runs.")
    protocol = output / "protocol.json"
    if protocol.exists() and study.read(protocol) != spec:
        raise ValueError("the saved passive-continuation protocol differs")
    if not protocol.exists():
        frozen.mkdir(parents=True, exist_ok=True)
        for path in Path(original["frozen_method_source"]).glob("*.py"):
            shutil.copyfile(path, frozen / path.name)
        planner = frozen / "validation_continuation.py"
        lines = planner.read_text().splitlines(keepends=True)
        node = next(item for item in ast.parse("".join(lines)).body
                    if isinstance(item, ast.FunctionDef) and item.name == "plan_validation_round")
        lines[node.body[0].lineno - 1:node.end_lineno] = [
            '    """Experimental ablation: retain passive evidence without active quotes."""\n',
            "    return None\n"]
        planner.write_text("".join(lines))
        objective = frozen / "objective_research.py"
        lines = objective.read_text().splitlines(keepends=True)
        node = next(item for item in ast.walk(ast.parse("".join(lines)))
                    if isinstance(item, ast.FunctionDef) and item.name == "_research_scope")
        lines[node.body[0].end_lineno:node.body[0].end_lineno] = [
            '        if research["phase"] == "validating" and len(research["validation"]) >= 4:\n',
            "            return None\n"]
        objective.write_text("".join(lines))

    baseline, full = comparison_references(source, original)
    research = {}
    measured_source = Path(original["source_study"])
    evaluations = set()
    for case in cases:
        key = case["research_id"]
        if key not in research:
            path = measured_source / "research" / f"{key}.json"
            saved = study.read(path)
            if (not saved.get("optimizer_initial_environment_state")
                    or saved["result"]["costs"]["observation_cost"]
                    + saved.get("supplemental_initial_calls", 0) != 32):
                raise ValueError("passive replay requires the already paid complete 32-row initialization")
            saved.update(research_origin=str(path.resolve()), actual_new_initial_calls=0,
                         actual_new_research_calls=0)
            research[key] = saved
        name = f"{case['next_problem']}_{case['task']['seed']}_{case['task']['noise']:.2f}.json"
        if name not in evaluations:
            original_evaluation = source / "evaluation" / name
            study.read(original_evaluation)
            destination = output / "evaluation" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(original_evaluation, destination)
            evaluations.add(name)
    study.write(protocol, spec)
    study.write(output / "measurement_references.json", {
        "research": {key: row["research_origin"] for key, row in research.items()},
        "evaluation": [str((source / "evaluation" / name).resolve()) for name in sorted(evaluations)]})
    return spec, research, baseline, full


def summarize(spec, output, results, baseline, full):
    passive = {row["case_id"]: row for row in results}
    cases = spec["cases"]
    mappings = {validation.ARMS[0]: baseline, ARM: passive, validation.ARMS[1]: full}
    brief = {"protocol": str((output / "protocol.json").resolve()), "cases": len(cases),
        "actual_new_optimizer_runs": len(results),
        "actual_new_online_calls": sum(row["costs"]["optimizer_new_simulator_calls"] for row in results),
        "actual_new_initial_calls": 0, "actual_new_evaluation_calls": 0,
        "comparisons": {
            "baseline_to_passive": validation.reuse.comparison(cases, baseline, passive),
            "passive_to_full": validation.reuse.comparison(cases, passive, full),
            "baseline_to_full": validation.reuse.comparison(cases, baseline, full)},
        "objective_research_compute": {arm: sum((mapping[case["case_id"]].get("objective_research") or {})
            .get("compute_cost", 0) for case in cases) for arm, mapping in mappings.items()},
        "behavior": [{"case_id": case["case_id"], **{arm: validation.objective_behavior(
            mapping[case["case_id"]]) for arm, mapping in mappings.items()}} for case in cases],
        "interpretation": "Each comparison is [left, right]; improvement means the right arm wins. "
            "The complete active continuation pool scan and scenario quote are ablated. Candidate retention, shared ordinary "
            "labels and completed validation rounds remain available. These are development cases; "
            "run sequentially after the full study to avoid timing interference."}
    study.write(output / "result_brief.json", brief)
    study.write(output / "summary.json", {"protocol": spec, "result_references": [
        {"case_id": row["case_id"], "arm": ARM,
         "case_record": str((output / "runs" / f"{row['case_id']}_{ARM}.json").resolve())}
        for row in results]})
    print(json.dumps({"cases": len(cases), "new_online_calls": brief["actual_new_online_calls"],
        "comparisons": {name: item["feasibility_first"] for name, item in brief["comparisons"].items()}}),
        flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "run"), default="plan")
    args = parser.parse_args()
    spec, research, baseline, full = prepare(args.source_study, args.output)
    if args.phase == "plan":
        print(json.dumps({"status": "prepared_not_run", "cases": len(spec["cases"]),
            "new_online_calls": spec["expected_new_online_calls"], "new_initial_calls": 0,
            "new_evaluation_calls": 0, "research_groups_reused": len(research),
            "protocol": str((args.output / "protocol.json").resolve())}))
        return
    module, policy, recovery = validation.arm_modules(spec)[validation.ARMS[1]]
    study.ARMS = (ARM,)
    study.controlled_problem = validation.active.framework.goal.controlled_problem
    results = study.optimize(spec, args.output, spec["cases"], research,
        arm_modules={ARM: (module, policy, recovery)})
    study.evaluate(spec, args.output, spec["cases"], results,
        environment_factory=validation.active.framework.goal.evaluation_environment)
    for row in results:
        study.write(args.output / "runs" / f"{row['case_id']}_{ARM}.json", row)
    summarize(spec, args.output, results, baseline, full)


if __name__ == "__main__":
    main()
