"""A development-only scope intervention, retaining the actual0.37 goal policy."""
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_sc_kg_research_scope_study as experiment

study, accounting, repair, goal = experiment.study, experiment.accounting, experiment.repair, experiment.goal
source = ROOT / "artifacts/sc_kg_objective_exploration_v037_20260910"
output = Path(__file__).resolve().parent / "scope_only_development"
arms = ("frozen_v037_objective_exploration", "scope_revision_only_v038")
experiment.ARMS = study.ARMS = accounting.ARMS = repair.ARMS = goal.ARMS = arms
study.controlled_problem = goal.controlled_problem
spec = deepcopy(study.read(source / "protocol.json"))
case = deepcopy(next(item for item in spec["cases"]
                     if item["case_id"] == "new_confirmation_2_mechanism_change_noise_0.03"))
case["split"] = "development"
spec.update(experiment="scope_only_development_with_saved_v037_goal_policy", cases=[case],
    arms=list(arms), development_source=str(source.resolve()),
    historical_baseline_arm="objective_exploration", historical_baseline_origin="historical_v037_objective_exploration",
    frozen_baseline_source=str((output.parent / "baseline_source_v037").resolve()),
    new_method={"status": "development_only_separate_from_final_joint_method",
        "intervention": "Use the saved0.37 SC adapter, objective research and adoption policy with the current JointModelRevision controller. Preserve the previously failing goal trajectory until the research-scope intervention.",
        "scope_rule": "During research, four distinct actual controls jointly predicted within the frozen threshold establish local predictive recovery. A subsequent joint anomaly requests same-point confirmation. If persistent, preserve the previous episode and start a new evidence scope at that anomaly; otherwise resume the interrupted frozen plan.",
        "budget": "N64 remains unchanged. This is one additional development optimizer run,32 online calls, reported separately and never reused as the final joint-method arm."})
study.write(output / "protocol.json", spec)
accounting.prepare_reuse(spec, output)
modules = experiment.arm_modules(spec)
baseline_module, baseline_policy, _ = modules[arms[0]]
from hegel_machine.cognition.joint_model_revision import JointModelRevision
modules[arms[1]] = (baseline_module, baseline_policy, JointModelRevision)
research = study.acquire_research(spec, output, [case])
results = study.optimize(spec, output, [case], research, arm_modules=modules)
evaluations = study.evaluate(spec, output, [case], results)
accounting.summarize(spec, output, [case], research, results, evaluations, "all")
repair.enrich_repair_summary(spec, output, "all")
goal.enrich_objective_summary(spec, output, "all")
