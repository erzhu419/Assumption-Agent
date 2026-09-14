#!/usr/bin/env python3
"""Reserve paid observations for finishing terminal research questions."""
from copy import deepcopy
import importlib
from pathlib import Path
import sys
import types

import run_sc_kg_supported_exploration_study as previous

framework, study = previous.framework, previous.study
ARMS = ("frozen_v042_supported_exploration", "completion_value")
SEEDS = (2043011203, 2043111217, 2043211231)
CONTROL_CASES = ("forward_1_mechanism_change_noise_0.03", "research_scope_3_stable_rzdt2_noise_0.00")
REPLAY_CASES = ("objective_only_change_3_noise_0.03",
    "research_scope_3_objective_only_change_noise_0.03", "joint_evidence_1_objective_only_change_noise_0.03",
    "research_scope_2_objective_only_change_noise_0.03", "objective_only_change_1_noise_0.03",
    "active_confirmation_2_mechanism_change_noise_0.03", "supported_exploration_1_objective_only_change_noise_0.03",
    "forward_1_mechanism_change_noise_0.00")


def protocol(source, output):
    spec = deepcopy(study.read(source / "protocol.json"))
    for key in ("diagnostic_replay_path", "previous_precision_attempt_path", "expected_cumulative_optimizer_runs"):
        spec.pop(key, None)
    spec.update(experiment="terminal_research_completion_budget", development_source=str(source.resolve()),
        arms=list(ARMS), frozen_baseline_source=str((output / "baseline_source_v042").resolve()),
        historical_baseline_arm="supported_exploration", historical_baseline_origin="historical_v042_supported_exploration")
    spec["baseline_method"] = spec.pop("new_method")
    first_seed = next(c["task"]["seed"] for c in spec["cases"] if c["split"] == "confirmation")
    templates = [deepcopy(c) for c in spec["cases"] if c["task"]["seed"] == first_seed and c["split"] in ("confirmation", "stress")]
    if len(templates) != 9:
        raise ValueError("Each new seed requires the prespecified six ordinary and three stress profiles")
    for case in spec["cases"]:
        case["previous_split"], case["split"] = case["split"], "development"
    for index, seed in enumerate(SEEDS, 1):
        for template in templates:
            case = deepcopy(template)
            case["task"]["seed"] = seed
            noise, problem = case["task"]["noise"], case["task"]["problem"]
            case["case_id"] = f"completion_value_{index}_{case['profile']}_noise_{noise:.2f}"
            case["research_id"] = f"completion_value_{index}_{problem}_noise_{noise:.2f}"
            spec["cases"].append(case)
    entries = study.read(output / "entry_inventory.json")["entries"]
    spec["affected_development_cases"] = [r["case_id"] for r in entries if r["head_stages"]]
    spec["unchanged_entry_reuse"] = [r["case_id"] for r in entries if not r["head_stages"]]
    spec["actual_control_cases"], spec["reused_control_cases"] = [], list(CONTROL_CASES)
    spec["actual_unchanged_control"] = CONTROL_CASES[1]
    spec["expected_new_optimizer_runs"] = len(spec["affected_development_cases"]) + 54
    spec["new_method"] = study.read(output / "method.json")
    spec["baseline_scope"] = "The 301 executed final v0.42 runs are development comparators. Reuse all old paid initial vectors, independent evaluation and original negative outcomes."
    spec["scope"] = "Three unused seeds each produce six ordinary confirmation profiles and three noisy stress profiles: one outlier, outlier followed by change, and objective-only change. The roster and completion method are frozen before their data are acquired."
    spec["reuse_accounting"] = "Rerun 84 old installed-head cases, reuse 217 no-head cases, and run both Hegel arms on 27 fresh cases: 138 actual runs/4416 online calls. The eight development runs are part of these 84, never repeated. Fresh initial384 and evaluation1920 bring the Hegel study to6720 physical calls. A separately accounted two-vanilla fresh comparison costs2592 additional calls; combined9312."
    return spec


def prepare_unaffected_reuse(spec, output):
    source = Path(spec["development_source"])
    for key in spec["unchanged_entry_reuse"]:
        path = output / "runs" / f"{key}_{ARMS[1]}.json"
        if path.exists():
            continue
        original = source / "runs" / f"{key}_supported_exploration.json"
        row = study.read(original)
        row.update(arm=ARMS[1], split="development", run_origin=spec["historical_baseline_origin"],
            original_result=str(original.resolve()), reuse_reason="No installed objective head; the modified terminal completion-planning path is never reached.")
        study.write(path, row)


def arm_modules(spec):
    name = "hegel_machine.cognition._completion_baseline_v042"
    package = types.ModuleType(name)
    package.__path__ = [spec["frozen_baseline_source"], str(study.ROOT / "src/hegel_machine/cognition")]
    sys.modules[name] = package
    baseline = {item: importlib.import_module(f"{name}.{item}") for item in
        ("sc_kg_environment", "prior_adoption", "joint_model_revision")}
    method_name = "hegel_machine.cognition._completion_method_v043"
    method_package = types.ModuleType(method_name)
    method_package.__path__ = [str(Path(spec["frozen_baseline_source"]).parent / "method_source_v043"),
        str(study.ROOT / "src/hegel_machine/cognition")]
    sys.modules[method_name] = method_package
    method = {item: importlib.import_module(f"{method_name}.{item}") for item in
        ("sc_kg_environment", "prior_adoption", "joint_model_revision")}
    study.MetaPriorResearch = importlib.import_module(f"{name}.meta_prior_research").MetaPriorResearch
    return {ARMS[0]:(baseline["sc_kg_environment"],baseline["prior_adoption"].PriorAdoption,baseline["joint_model_revision"].JointModelRevision),
        ARMS[1]:(method["sc_kg_environment"],method["prior_adoption"].PriorAdoption,
            method["joint_model_revision"].JointModelRevision)}


def enrich_summary(spec, output, phase):
    previous._enrich_precision(spec, output, phase)
    stem = "" if phase == "all" else f"{phase}_"
    brief = study.read(output / f"{stem}result_brief.json")
    previous.add_outcome_groups(brief)
    compact = study.read(output / f"{stem}compact_summary.json")
    summary = study.read(output / f"{stem}summary.json")
    def metrics(rows):
        plans = [p for r in rows for p in (r.get("objective_research") or {}).get("completion_plans", [])]
        requests = [q for r in rows for q in r["costs"].get("objective_diagnostics", {}).get("requests", [])]
        return {"planning_rounds":len(plans),
            "deferred_nominations":sum(len(p["deferred"]) for p in plans),
            "deadline_nominations":sum(o.get("completion_plan", {}).get("must_complete_now", False) for p in plans for o in p["offers"]),
            "paid_deadline_actions":sum(q.get("completion_plan", {}).get("must_complete_now", False) for q in requests),
            "posterior_updates":sum(r["costs"].get("objective_diagnostics", {}).get("head_posterior_updates",0) for r in rows)}
    actual = [r for r in summary["results"] if r.get("run_origin") != spec["historical_baseline_origin"]]
    brief["actual_new_completion_plans"] = metrics(actual)
    by_case = {r["case_id"]:r for r in summary["results"] if r["arm"] == ARMS[1]}
    for case in brief["cases"]:
        case["completion_plans"] = metrics([by_case[case["case_id"]]])
    for group in compact["groups"]:
        for arm in ARMS:
            group[arm]["completion_plans"] = metrics([r for r in summary["results"] if
                (r["split"],r["profile"],r["task"]["noise"],r["arm"]) ==
                (group["split"],group["profile"],group["noise"],arm)])
    compact["current_study_accounting"] = spec["reuse_accounting"]
    compact["actual_new_completion_plans"] = brief["actual_new_completion_plans"]
    study.write(output / f"{stem}result_brief.json",brief)
    study.write(output / f"{stem}compact_summary.json",compact)


def main():
    framework.ARMS, framework.REPLAY_CASES, framework.CONTROL_CASES = ARMS, REPLAY_CASES, CONTROL_CASES
    framework.protocol, framework.arm_modules = protocol, arm_modules
    framework.prepare_unaffected_reuse, framework.enrich_precision_summary = prepare_unaffected_reuse, enrich_summary
    framework.main()


if __name__ == "__main__":
    main()
