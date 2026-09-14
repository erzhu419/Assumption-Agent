#!/usr/bin/env python3
"""One frozen real-simulator change case for evidence-based prior withdrawal."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.meta_prior_research import MetaPriorResearch
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.prior_adoption import PriorAdoption
from hegel_machine.cognition.sc_kg_environment import (
    HegelProgramProblem, SCKGEnvironment, initialize_sc_kg_optimizer,
)


SEED = 1043010067
CHANGE_AFTER_NEW_CALLS = 12


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


class MechanismChangeProblem(HegelProgramProblem):
    """Change the actual simulator, without informing the adoption policy."""

    def __init__(self, environment, program, *, change_after=CHANGE_AFTER_NEW_CALLS):
        super().__init__(environment, program)
        from problems.rzdt import RZDT2
        self.next_problem = RZDT2(d=environment._problem.d, L=100,
            sigma=environment._settings["noise"], heteroscedastic=False)
        self.change_after = change_after
        self.simulation_log = []

    def simulate(self, point, rng=None):
        import numpy as np
        from problems.single_objective import ScalarizedProblem
        if self.simulator_calls == self.change_after:
            self.base = ScalarizedProblem(self.next_problem)
        vector = [float(value) for value in self.base.base.simulate(point, rng)]
        output = [float(self.base.weights[0] * vector[0] + self.base.weights[1] * vector[1]),
                  vector[2]]
        self.simulator_calls += 1
        self.simulation_log.append({"new_call": self.simulator_calls,
            "mechanism": "state_policy_rzdt1" if self.simulator_calls <= self.change_after else "rzdt2",
            "point": list(map(int, point)), "vector_outputs": vector, "output": output})
        return np.asarray(output, dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.source_study / "protocol.json").read_text())
    task = {"problem": "state_policy_rzdt1", "dimension": 3, "seed": SEED, "noise": 0.}
    config = replace(ResearchConfig(**source["research_config"]), seed=SEED)
    protocol = {"experiment": "single_real_mechanism_change_prior_adoption",
        "task": task, "change_after_new_calls": CHANGE_AFTER_NEW_CALLS,
        "next_problem": "rzdt2", "total_calls": 64,
        "research_config": asdict(config), "optimizer_settings": source["optimizer_settings"],
        "optimizer_seed": SEED + source["optimizer_seed_offset"],
        "policy": source["policy"], "sc_kg_project_root": source["sc_kg_project_root"],
        "evidence": "A fresh StatePolicyRZDT1 research run acquires 32 real vector observations. The next 12 actual optimizer calls still use StatePolicyRZDT1, then all remaining calls use the existing RZDT2 simulator. The adoption policy sees only paired pre-measurement predictions, the measured target and noise estimate; no mechanism label is supplied.",
        "scope": "One preselected new seed, frozen policy and thresholds, no tuning after results. This is a functional mechanism-change case, not a performance comparison. Final truth scoring uses the final RZDT2 mechanism after the recommendation freezes."}
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("the saved mechanism-change protocol differs")
    write(protocol_path, protocol)
    environment = SCKGEnvironment(source["sc_kg_project_root"], **task)
    research_path = args.output / "research.json"
    if research_path.exists():
        saved = json.loads(research_path.read_text())
        report = saved["result"]
        environment.restore(report["research_state"]["environment_state"])
    else:
        library = KnowledgeLibrary()
        report = MetaPriorResearch(config, library).run(environment)
        write(research_path, {"task": task, "result": report, "library": library.to_dict()})
    if report["costs"]["observation_cost"] != 32 or not report.get("program"):
        raise RuntimeError("the frozen case needs 32 paid observations and a learned executable program")
    adapter = MechanismChangeProblem(environment, Program.from_dict(report["program"]))
    from algorithms.single_olhkg import SingleOLHKGConfig
    optimizer_config = SingleOLHKGConfig(N=64, seed=protocol["optimizer_seed"],
                                        **protocol["optimizer_settings"])
    policy = PriorAdoption(minimum_gain=source["policy"]["minimum_gain"],
                           batch_size=source["policy"]["minimum_new_observations"])
    algorithm = initialize_sc_kg_optimizer(adapter, optimizer_config, adoption_policy=policy)
    final = algorithm.run()
    events = algorithm.adoption_events
    adopted = [event["stage"] for event in events
               if event["selected_before"] == "default" and event["selected_after"] == "learned"]
    withdrawn = [event["stage"] for event in events
                 if event["selected_before"] == "learned" and event["selected_after"] == "default"]
    phases = {}
    predictions = policy.to_dict()["observations"]
    for name, rows in (("before_change", predictions[:CHANGE_AFTER_NEW_CALLS]),
                       ("after_change", predictions[CHANGE_AFTER_NEW_CALLS:])):
        phases[name] = {"new_observations": len(rows),
                       "default_mse": mean(row["default_error"] for row in rows),
                       "learned_mse": mean(row["learned_error"] for row in rows)}
    summary = {"protocol": protocol, "research_status": report["status"],
        "research_family": report.get("family_id"), "program": report["program"],
        "costs": {"research_simulator_calls": adapter.research_simulator_calls,
                  "research_compute": report["costs"]["compute_cost"],
                  "optimizer_new_simulator_calls": adapter.simulator_calls,
                  "total_simulator_calls": adapter.research_simulator_calls + adapter.simulator_calls,
                  "adoption_compute": algorithm.adoption_diagnostics},
        "loaded_observation_count": adapter.loaded_observation_count,
        "optimizer_history_count": len(algorithm.history),
        "policy": policy.to_dict(), "events": events,
        "simulations": adapter.simulation_log,
        "adopted_at_stages": adopted, "withdrawn_at_stages": withdrawn,
        "adopted_then_withdrawn": any(retract > adopt for adopt in adopted for retract in withdrawn),
        "phase_prediction_error": phases,
        "active_constraint_basis": algorithm.active_constraint_basis,
        "actions": [{key: row.get(key) for key in
            ("stage", "x_selected", "Y_observed", "mu_before", "sigma2_before")}
            for row in algorithm.iteration_log],
        "final": {key: final.get(key) for key in
            ("x_recommended", "true_objective", "true_constraint_mean", "true_feasible", "n_simulations")}}
    write(args.output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in (
        "research_status", "research_family", "costs", "adopted_at_stages",
        "withdrawn_at_stages", "adopted_then_withdrawn", "active_constraint_basis")}), flush=True)


if __name__ == "__main__":
    main()
