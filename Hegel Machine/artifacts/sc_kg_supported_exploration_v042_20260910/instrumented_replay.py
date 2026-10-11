"""One frozen v0.41 replay; copy already-computed terminal screening evidence."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from run_sc_kg_objective_exploration_study import controlled_problem
from hegel_machine.cognition.language import Program

source = ROOT / "artifacts/sc_kg_value_precision_v041_20260910"
protocol = json.loads((source / "protocol.json").read_text())
case_id = "research_scope_2_objective_only_change_noise_0.03"
case = next(item for item in protocol["cases"] if item["case_id"] == case_id)
initial = json.loads((source / "research" / f"{case['research_id']}.json").read_text())
saved = json.loads((source / "runs" / f"{case_id}_value_precision.json").read_text())
package_name = "hegel_machine.cognition._instrumented_v041"
package = types.ModuleType(package_name)
package.__path__ = [str(OUT / "baseline_source_v041"), str(ROOT / "src/hegel_machine/cognition")]
sys.modules[package_name] = package
sc, adoption, recovery, objective = [importlib.import_module(f"{package_name}.{name}")
    for name in ("sc_kg_environment", "prior_adoption", "joint_model_revision", "objective_research")]
original = objective.ObjectiveResearch._exploration_offer
screening = []

def observed_offer(self, rows, grouped, incumbent, predict, budget, stage, *, quantized=False):
    record = {"stage": stage + 1, "incumbent": deepcopy(incumbent),
              "budget_before": budget.remaining, "predictions": []}

    def observed_prediction(point):
        prediction = predict(point)
        if stage >= 60:
            record["predictions"].append({"requested_features": deepcopy(point),
                "prediction": deepcopy(prediction),
                "passes_feasible": prediction["feasible"],
                "passes_objective": prediction.get("objective_mean", float("inf")) < incumbent["objective_mean"]})
        return prediction

    result = original(self, rows, grouped, incumbent, observed_prediction, budget, stage, quantized=quantized)
    if stage >= 60:
        record.update(selected=deepcopy(result), budget_after=budget.remaining)
        screening.append(record)
    return result

objective.ObjectiveResearch._exploration_offer = observed_offer
environment = sc.SCKGEnvironment(protocol["sc_kg_project_root"], **case["task"])
environment.restore(initial["optimizer_initial_environment_state"])
adapter = controlled_problem(sc.HegelProgramProblem, environment,
    Program.from_dict(initial["result"]["program"]), case)
from algorithms.single_olhkg import SingleOLHKGConfig
config = SingleOLHKGConfig(N=protocol["total_calls_per_arm"],
    seed=case["task"]["seed"] + protocol["optimizer_seed_offset"], **protocol["optimizer_settings"])
joint = recovery.JointModelRevision(environment.spec,
    minimum_error=protocol["revision"]["minimum_error"], batch_size=protocol["revision"]["batch_size"],
    seed=case["task"]["seed"] + protocol["revision"]["seed_offset"], compute_budget=protocol["revision"]["compute_budget"])
algorithm = sc.initialize_sc_kg_optimizer(adapter, config,
    adoption_policy=adoption.PriorAdoption(minimum_gain=protocol["policy"]["minimum_gain"],
        batch_size=protocol["policy"]["minimum_new_observations"]), recovery=joint)
started = time.perf_counter()
final = algorithm.run()
elapsed = time.perf_counter() - started
paid = [{"point": list(point), "output": [float(v) for v in output]} for point, output in algorithm.history[32:]]
expected = [{"point": item["point"], "output": item["output"]} for item in saved["simulations"]]
result = {"case_id": case_id, "frozen_source": str(OUT / "baseline_source_v041"),
    "same_paid_trajectory": paid == expected, "final_recommendation": list(final["x_recommended"]),
    "initial_paid_vectors_reused": 32, "new_physical_calls": adapter.simulator_calls,
    "extra_diagnostic_prediction_calls": 0, "replay_seconds": elapsed,
    "objective_compute": algorithm.objective_research.state["compute_cost"],
    "recovery_costs": joint.to_dict()["costs"],
    "value_precision": algorithm.objective_diagnostics["value_precision"],
    "paid_observations": paid, "screening": screening}
target = OUT / "instrumented_baseline_replay.json"
target.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: result[key] for key in ("same_paid_trajectory", "final_recommendation",
    "new_physical_calls", "extra_diagnostic_prediction_calls", "objective_compute", "replay_seconds")}))
print(target)
