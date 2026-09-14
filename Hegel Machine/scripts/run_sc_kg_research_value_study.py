#!/usr/bin/env python3
"""Compare short research sequences inside the original terminal KG competition."""
from copy import deepcopy
import importlib
from pathlib import Path
import sys
import types

import run_sc_kg_supported_exploration_study as previous

framework, study = previous.framework, previous.study
ARMS = ("frozen_v042_supported_exploration", "research_value")
SEEDS = (1343011307, 1343111321, 1343211339)
CONTROL_CASES = ("forward_1_mechanism_change_noise_0.03", "research_scope_3_stable_rzdt2_noise_0.00")
REPLAY_CASES = ('research_scope_3_objective_only_change_noise_0.03', 'research_scope_2_objective_only_change_noise_0.03', 'completion_value_1_objective_only_change_noise_0.03', 'objective_only_change_1_noise_0.03', 'active_confirmation_1_stable_rzdt2_noise_0.00', 'forward_1_mechanism_change_noise_0.00', 'active_confirmation_2_mechanism_change_noise_0.03', 'research_scope_3_stable_rzdt2_noise_0.00')


def protocol(source, output):
    spec = deepcopy(study.read(source / "protocol.json"))
    for key in ("diagnostic_replay_path", "previous_precision_attempt_path", "expected_cumulative_optimizer_runs"):
        spec.pop(key, None)
    spec.update(experiment="terminal_research_sequence_value", development_source=str(source.resolve()),
        arms=list(ARMS), frozen_baseline_source=str((output / "baseline_source_default_v043").resolve()),
        historical_baseline_arm="frozen_v042_supported_exploration", historical_baseline_origin="historical_v042_supported_exploration")
    spec.pop("new_method", None)
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
            case["case_id"] = f"research_value_{index}_{case['profile']}_noise_{noise:.2f}"
            case["research_id"] = f"research_value_{index}_{problem}_noise_{noise:.2f}"
            spec["cases"].append(case)
    entries = study.read(output / "entry_inventory.json")["entries"]
    spec["affected_development_cases"] = [r["case_id"] for r in entries if r["terminal_offer_stages"]]
    spec["unchanged_entry_reuse"] = [r["case_id"] for r in entries if not r["terminal_offer_stages"]]
    spec["actual_control_cases"], spec["reused_control_cases"] = [], list(CONTROL_CASES)
    spec["actual_unchanged_control"] = CONTROL_CASES[1]
    spec["expected_new_optimizer_runs"] = len(spec["affected_development_cases"]) + 54
    spec["new_method"] = study.read(output / "method.json")
    spec["baseline_scope"] = "The 328 executed default v0.42 runs are development comparators. Reuse all old paid initial vectors, independent evaluation and original negative outcomes."
    spec["scope"] = "Three unused seeds each produce six ordinary confirmation profiles and three noisy stress profiles: one outlier, outlier followed by change, and objective-only change. The roster and research-sequence method are frozen before their data are acquired."
    spec["reuse_accounting"] = "Rerun88 old terminal-offer cases and reuse240 paths without terminal offers; run both Hegel arms on27 fresh cases:142 actual optimizer runs/4544 online. The seven development runs are part of88, never repeated; one no-head control is reused. Fresh initial384 and evaluation1920 bring Hegel study to6848 physical calls. Fresh shared vanilla864 and cold native1728 bring the combined total9440."
    return spec


def prepare_unaffected_reuse(spec, output):
    source = Path(spec["development_source"])
    for key in spec["unchanged_entry_reuse"]:
        path = output / "runs" / f"{key}_{ARMS[1]}.json"
        if path.exists():
            continue
        original = source / "runs" / f"{key}_{spec['historical_baseline_arm']}.json"
        row = study.read(original)
        row.update(arm=ARMS[1], split="development", run_origin=spec["historical_baseline_origin"],
            original_result=str(original.resolve()), reuse_reason="No actual terminal offers; the research-sequence path is never reached.")
        study.write(path, row)


def arm_modules(spec):
    name = "hegel_machine.cognition._research_value_baseline_v042"
    package = types.ModuleType(name)
    package.__path__ = [spec["frozen_baseline_source"], str(study.ROOT / "src/hegel_machine/cognition")]
    sys.modules[name] = package
    baseline = {item: importlib.import_module(f"{name}.{item}") for item in
        ("sc_kg_environment", "prior_adoption", "joint_model_revision")}
    method_name = "hegel_machine.cognition._research_value_method_v044"
    method_package = types.ModuleType(method_name)
    method_package.__path__ = [str(Path(spec["frozen_baseline_source"]).parent / "method_source_v044"),
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
        rounds = [p for r in rows for p in r["costs"].get("objective_diagnostics", {}).get("research_value", {}).get("rounds", [])]
        return {"rounds":len(rounds), "supported_sequences":sum(p["status"] == "supported_sequence" for p in rounds),
            "changed_original_action":sum(p.get("selected_point") is not None and p["selected_point"] != p["baseline_point"] for p in rounds),
            "unresolved_rounds":sum(p["status"] == "unresolved" for p in rounds),
            **{key:sum(p.get(key,0) for p in rounds) for key in ("evaluations","seconds","terminal_value_calls","terminal_pool_rows")},
            "planned_call_lengths":[p["planned_calls"] for p in rounds if p.get("selected_point") is not None]}
    actual = [r for r in summary["results"] if r.get("run_origin") != spec["historical_baseline_origin"]]
    brief["actual_new_research_sequences"] = metrics(actual)
    brief["fresh_research_sequences"] = {arm: metrics([r for r in actual if r["arm"] == arm and r["split"] != "development"]) for arm in ARMS}
    brief["research_sequence_accounting"] = "Sequence evaluations and seconds are subsets of the shared value_precision totals and complete native KG time. Each evaluation is one candidate/path/fantasy-step. A supported plan only changes the next real action; remaining steps are hypothetical, never prepaid."
    by_case = {r["case_id"]:r for r in summary["results"] if r["arm"] == ARMS[1]}
    for case in brief["cases"]:
        case["research_sequences"] = metrics([by_case[case["case_id"]]])
    for group in compact["groups"]:
        for arm in ARMS:
            group[arm]["research_sequences"] = metrics([r for r in summary["results"] if
                (r["split"],r["profile"],r["task"]["noise"],r["arm"]) ==
                (group["split"],group["profile"],group["noise"],arm)])
    compact["current_study_accounting"] = spec["reuse_accounting"]
    compact["actual_new_research_sequences"] = brief["actual_new_research_sequences"]
    compact["fresh_research_sequences"] = brief["fresh_research_sequences"]
    compact["research_sequence_accounting"] = brief["research_sequence_accounting"]
    study.write(output / f"{stem}result_brief.json",brief)
    study.write(output / f"{stem}compact_summary.json",compact)


def main():
    framework.ARMS, framework.REPLAY_CASES, framework.CONTROL_CASES = ARMS, REPLAY_CASES, CONTROL_CASES
    framework.protocol, framework.arm_modules = protocol, arm_modules
    framework.prepare_unaffected_reuse, framework.enrich_precision_summary = prepare_unaffected_reuse, enrich_summary
    framework.main()


if __name__ == "__main__":
    main()
