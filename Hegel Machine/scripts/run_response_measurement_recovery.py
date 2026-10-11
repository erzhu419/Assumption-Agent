#!/usr/bin/env python3
"""Run measurement recovery through the existing parent and physical controller.

The offline branch supplies one author-written scientific declaration after the
parent refutes its initial families. It tests execution and revision, not LLM
discovery. The codex_cli branch instead lets the existing child inquiry choose
its own measurement and hypothesis, without that declaration or extra hints.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]

from test_world_reference_hypothesis import ReferenceLab, prepared, square, submit, PATH
from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.world_reference_hypothesis import _retain_declaration
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json


class AliasedSignalLab(ReferenceLab):
    """Existing quantized task with a separately measurable latent signal."""

    def __init__(self, case="informative"):
        if case not in ("informative", "uninformative"):
            raise ValueError("unknown measurement fixture")
        super().__init__(intercept=100.3)
        self.signal = deepcopy(self.density)
        self.signal[5] = 395.5 ** .5
        self.required[5] = 4055.3
        self.density[2] = self.density[1]
        self.gap[2] = self.gap[1]
        if case == "uninformative":
            self.signal = deepcopy(self.density)

    def public_observation(self):
        view = super().public_observation()
        if "dialogIn" in view["dialog_box"]:
            view["dialog_box"]["dialogIn"] += " The allowable range is 0 to 10,000 Hz."
        return view

    def act(self, action):
        result = super().act(action)
        if action.get("chosen_dialog_option_int") not in (None, 1):
            # Match the public 0..10000 control range and DiscoveryWorld's
            # DialogTree: apply the requested increment, then clamp its result.
            number = self.dialog_number
            self.frequencies[number] = min(10000., max(0., self.frequencies[number]))
            result["observation"] = self.public_observation()
        if action.get("action") == "USE" and action["arg1"] == 93:
            number = action["arg2"] - 200
            self.last_message = ("You examine quantum crystal " + str(number) +
                ". The signal magnitude = " + str(self.signal[number]) + " units.")
            result["observation"] = self.public_observation()
        return result


def prepared_conflict(workspace, case="informative"):
    """Buy the entire failing prefix with the unchanged parent procedure."""
    workspace = Path(workspace)
    agent, _ = prepared(workspace)
    lab = AliasedSignalLab(case)
    agent.lab = lab
    source = (workspace / PATH).read_text()
    for feature in ("density", "gap"):
        source = _retain_declaration(source, square(feature, feature + "_square"))
    agent._execute("write", {"path": PATH, "code": source}, lab)
    execution = agent._run_procedure(lab, PATH, max_steps=256)
    state = json.loads((workspace / recipe.STATE).read_text())
    return agent, lab, execution, state


def signal_hypothesis():
    hypothesis = square("signal", "signal_square")
    hypothesis["measurements"] = {"signal": {"instrument": "signal meter",
        "pattern": r"signal magnitude = ([\d.]+) units", "group": 1}}
    return hypothesis


def _snapshot(agent, lab, execution, state):
    return {"execution": deepcopy(execution), "state": deepcopy(state),
            "agent": agent.result(), "observations": deepcopy(agent.observations),
            "actions": deepcopy(lab.actions), "frequencies": deepcopy(lab.frequencies)}


def _brief(prefix, final, backend):
    before = len(prefix["observations"])
    state, result = final["state"], final["agent"]
    questions = result["research"]["questions"]
    sensor = [row["index"] for row in final["observations"][before:]
              if row["action"].get("action") == "USE" and row["action"].get("arg1") == 93]
    points = [deepcopy(q) for q in questions if q["id"].startswith("reference_test_")
              and q.get("status") == "refuted" and q.get("evidence")
              and min(q["evidence"]) >= before]
    physical = [deepcopy(q) for q in questions if q["id"].startswith("reference_response_")
                and q["id"].endswith("_physical") and q.get("evidence")
                and min(q["evidence"]) >= before]
    family = state.get("response_research", {}).get("family_research", {})
    return {"backend": backend, "study_status": "finished",
        "candidate_origin": ("author_supplied_after_refutation" if backend == "offline"
                             else "existing_live_inquiry_without_supplied_candidate"),
        "completed_successfully": result["score"]["completedSuccessfully"],
        "prefix_actions": len(prefix["actions"]), "total_actions": len(final["actions"]),
        "new_actions": len(final["actions"]) - len(prefix["actions"]),
        "prefix_family_statuses": prefix["state"].get("response_research", {}).get("family_research", {}).get("families", []),
        "prefix_research_obligation": prefix["state"].get("response_research", {}).get("family_research", {}).get("research_obligation"),
        "new_sensor_observation_indices": sensor, "new_point_refutations": points,
        "new_boolean_physical_answers": physical, "final_frequencies": final["frequencies"],
        "final_family_selection_role": family.get("selected_role"),
        "final_nonlocal_progress": family.get("nonlocal_progress"),
        "costs": result["costs"],
        "scope": "Actual controller recovery and physical support for tested predictions; "
                 "operational exploration does not establish family-wide target unanimity."}


def run(output, *, backend="offline", calls=32, case="informative"):
    if backend not in ("offline", "codex_cli"):
        raise ValueError("backend must be offline or codex_cli")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    auth = verify_codex_subscription() if backend == "codex_cli" else None
    protocol = {"study": "measurement_recovery_in_existing_reference_controller", "backend": backend, "case": case,
        "model": {"requested_alias": "gpt-6-luna", "reasoning_effort": "medium"} if backend == "codex_cli" else None,
        "auth_mode": auth, "max_model_calls": calls if backend == "codex_cli" else 0,
        "scope": "A development integration fixture, not a DiscoveryWorld benchmark. "
                 "The parent buys all prefix and recovery observations. Offline recovery receives "
                 "an author-written declaration only after actual family refutation. Live recovery "
                 "uses the parent's existing inquiry without a supplied scientific declaration."}
    write_json(output / "protocol.json", protocol)
    agent, lab, execution, state = prepared_conflict(output / "agent", case)
    prefix = _snapshot(agent, lab, execution, state)
    write_json(output / "prefix.json", prefix)
    protocol["prefix_limits"] = deepcopy(agent.limits)
    if backend == "codex_cli":
        agent.limits.update(max_model_calls=calls, task_wall_seconds=3600,
            provider_request_timeout_seconds=180, python_seconds_per_task=240)
    protocol["limits"] = deepcopy(agent.limits)
    write_json(output / "protocol.json", protocol)
    family = state.get("response_research", {}).get("family_research", {})
    obligation = family.get("research_obligation", {})
    if obligation.get("reason") != "declared_shared_response_families_refuted":
        execution = {"status": "prefix_did_not_reach_family_refutation"}
    elif backend == "offline":
        execution = submit(agent, lab, signal_hypothesis())
    else:
        agent.provider = CodexCodeProvider(agent.workspace, model="gpt-6-luna",
            reasoning_effort="medium", timeout_seconds=180)

        def checkpoint(data):
            write_json(output / "active.json", {**data, "study_status": "running"})
            print(json.dumps({"calls": agent.costs["model_calls"],
                "actions": lab.environment_steps, "completed": lab.success}), flush=True)

        agent._on_checkpoint = checkpoint
        execution = agent._run_procedure(lab, PATH, max_steps=256)
    state = json.loads((agent.workspace / recipe.STATE).read_text())
    final = _snapshot(agent, lab, execution, state)
    final.update(study_status="finished", backend=backend)
    write_json(output / "result.json", final)
    brief = _brief(prefix, final, backend)
    brief["case"] = case
    write_json(output / "brief.json", brief)
    return brief


def run_pair(output):
    """Fixed positive and uninformative instruments share controller and task."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", {"study": "paired_measurement_recovery", "backend": "offline",
        "cases": ["informative", "uninformative"], "scope": "Identical physical task, initial "
        "declarations and supplied recovery candidate. Only the signal meter's reading differs; "
        "the uninformative meter repeats the existing density reading."})
    results = {case: run(output / case, case=case) for case in ("informative", "uninformative")}
    write_json(output / "brief.json", results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("offline", "codex_cli"), default="offline")
    parser.add_argument("--calls", type=int, default=32)
    parser.add_argument("--case", choices=("informative", "uninformative", "pair"), default="informative")
    args = parser.parse_args()
    if args.case == "pair":
        if args.backend != "offline":
            parser.error("the paired integration diagnostic uses the offline backend")
        result = run_pair(args.output)
    else:
        result = run(args.output, backend=args.backend, calls=args.calls, case=args.case)
    print(json.dumps(result, indent=2))
