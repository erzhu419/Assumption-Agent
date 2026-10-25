#!/usr/bin/env python3
"""Let the same LLM use or reconstruct legacy response-experiment methods."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from random import Random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.code_agent import CodexCodeProvider, evaluate_python_model
from hegel_machine.cognition.online_inquiry_llm import COMMON_INTERFACE, InquiryLLMConfig
from hegel_machine.cognition.scale_bindings import bind_measurement_arguments
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab, _control_observation
from hegel_machine.cognition.temporal_instrument_research import temporal_instrument_program
from hegel_machine.cognition.temporal_method_lab import TemporalMethodLab
from hegel_machine.cognition.temporal_operator_agent import (
    INSTRUMENT_METHODS_PATH, OPERATOR_INTERFACE, TemporalOperatorAgent, export_temporal_program,
)
from hegel_machine.cognition.language import Program
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_temporal_method_learning import add_costs
from scripts import run_temporal_method_study as legacy_study


PILOT_INTERFACE = """
This episode uses an existing finite-hypothesis benchmark. hypotheses.json gives
the supplied response laws in the measured meter coordinate and the paths of their
controls-only Python predictors. They are alternatives, not accepted answers.
instrument.json was fitted on two paid 24-row meter trajectories and separately
checked on a third paid trajectory. These records and their costs are yours to
inspect. Candidate Python exports use this measured calibration, not a private
apparatus equation. Predict the response, including narrow activation and exit
events. Random observations may miss these events: a random confirmation pass
does not establish which candidate is correct. Terminal evaluation also includes
activation/exit trajectories, unavailable during research.

Choose experiments and a final predictor using the shared tools or ordinary
Python. No research actions are supplied as an executable policy. If a measured
experiment-method book is present, it is an available research resource; you may
use it, revise it or work without it. The goal is accurate prediction and useful
research, not a required operator sequence. Every step is chosen by you. Finish
with the selected Python predictor and a concise lesson. No research manifest is
required in this matched method-availability pilot. Source methods are created
only by actual measured operator feedback, never by the experiment runner.
"""


class LegacyMethodInquiryLab(TemporalInquiryLab):
    """Reuse the old fixture and its activation/exit terminal evaluation."""

    def __init__(self, case="window", stage=0, seed=151, *, center=.21,
                 half_width=1e-4, response_gain=2.):
        if stage not in (0, 4):
            raise ValueError("this pilot uses the original u/meter stages 0 and 4")
        super().__init__(task_index=1)
        self._laboratory = TemporalMethodLab(case, stage, seed, center=center,
            half_width=half_width, response_gain=response_gain)
        self.task_interface = self.task_interface.replace(
            "No mechanism formula or\nfinite hypothesis list is supplied. Learn and revise your own Python models.",
            "This pilot supplies competing response programs; select or revise them from paid evidence.")
        self.task_interface += "\n" + PILOT_INTERFACE + "\n" + OPERATOR_INTERFACE

    def evaluation_rows(self, *, length=128, seed=1000003):
        if length % 4 or length < 16:
            raise ValueError("legacy terminal evaluation uses four complete trajectories")
        rows = self._laboratory.evaluation_rows(seed=seed, trajectories=4, length=length // 4)
        return [_control_observation(row) for row in rows]


def protocol():
    limits = InquiryLLMConfig(observation_budget=240, confirmation_rows=48,
        max_model_calls=12, token_budget=400000, execution_seconds=120., compute_budget=100000)
    cases = []
    for case, seed, center in (("window", 151, -.31), ("event", 1154, .37)):
        cases.append({"case": case,
            "source": dict(case=case, stage=0, seed=seed, center=.21,
                           half_width=1e-4, response_gain=2.),
            "target": dict(case=case, stage=4, seed=seed + 222, center=center,
                           half_width=1e-4, response_gain=1.)})
    return {"study": "legacy_response_methods_same_llm_development", "model": "gpt-6-luna",
        "provider": {"backend": "codex_cli", "reasoning_effort": "medium", "timeout_seconds": 180},
        "cases": cases, "limits": asdict(limits), "conditions": ["retained", "removed"],
        "intervention": "Only the actually learned experiment-method JSON book differs. Both targets "
            "receive the same source lesson, current paid calibration, candidates, tools and limits.",
        "preparation": "Two 24-row meter trajectories fit the instrument; an independent 24-row "
            "trajectory assesses it. The original benchmark supplies gains 1 and 2 plus the null law; "
            "event cases also include the window alternative. Preparation is charged in every episode. "
            "No response experiment or final model is selected by the runner.",
        "transfer": "Freeze each source book before its target pair. Target feedback never reaches "
            "another target. Both arms may learn new methods using the same operators.",
        "terminal": "All source and target agents stop before the original fixture generates four "
            "terminal random/activation/exit trajectories. Report random fresh confirmation separately "
            "from unique candidate identification and terminal prediction.",
        "limitations": "Two matched development cases, supplied response candidates and paid initial "
            "calibration. This tests LLM consumption of the v0.76 method book, not autonomous candidate "
            "invention, stable average advantage, or the v0.92 instrument-extension capability."}


def prepare_episode(workspace, task, config, provider=None, book=None, lesson=""):
    files = ({INSTRUMENT_METHODS_PATH: json.dumps(book, indent=2)} if book is not None else {})
    agent = TemporalOperatorAgent(provider, workspace, arm="code_react", source_task=False,
        knowledge={"files": files, "lessons": [lesson] if lesson else []}, config=config,
        confirmation_seed=task["seed"] + 7919)
    # Keep the ordinary action/confirmation implementation. Persistence in this
    # intervention is exactly the supplied method file, not a three-arm protocol.
    agent.common_interface = COMMON_INTERFACE.split("For the hegel condition,")[0]
    lab = LegacyMethodInquiryLab(**task)

    def purchase(seed, batches):
        rng = Random(seed)
        return agent._execute("query", {"actions": [{"kind": "trajectory",
            "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(24)], "channels": ["meter"]}
            for _ in range(batches)]}, lab)["record_indices"]

    training = purchase(task["seed"] + 817, 2)
    agent._execute("run", {"operator": "instrument", "path": "instrument.json", "records": training}, lab)
    validation = purchase(task["seed"] + 981, 1)
    assessed = agent._execute("run", {"operator": "assess_instrument",
        "instrument": "instrument.json", "records": validation}, lab)
    instrument = temporal_instrument_program(json.loads((agent.workspace / "instrument.json").read_text()))
    if instrument is None or assessed["status"] != "supported_on_measured_trajectory":
        raise ValueError("the supplied paid calibration did not establish a unique instrument")
    legacy_study.Program = Program
    programs = legacy_study.candidate_programs(lab._laboratory, task["center"],
        {"half_width": task["half_width"], "candidate_gains": [1., 2.]})
    candidates = []
    for index, program in enumerate(programs):
        path = f"models/candidate_{index:02d}.py"
        executable = bind_measurement_arguments(program, {"meter": instrument})
        agent._save(path, export_temporal_program(executable))
        candidates.append({"hypothesis_id": path, "path": path, "program": program.to_dict()})
    agent._save("hypotheses.json", json.dumps(candidates, indent=2))
    return agent, lab


def episode_summary(result, terminal):
    operators = []
    for event in result["events"]:
        feedback = event.get("feedback", {})
        if feedback.get("operator"):
            operators.append(feedback)
        operators.extend(feedback.get("after_observation_results", []))
    assessments = [item for item in operators if item.get("operator") == "assess_response"
                   and "error" not in item]
    last = assessments[-1] if assessments else {}
    survivors = [item["hypothesis_id"] for item in last.get("hypotheses", [])
                 if item["status"] == "supported_on_probe"]
    identified = (last.get("status") == "distinguished" and len(survivors) == 1
                  and all(item["rows"] > 0 for item in last.get("hypotheses", [])))
    confirmed = bool((result.get("fresh_confirmation") or {}).get("passed"))
    correct = terminal.get("returncode") == 0 and terminal.get("mse", float("inf")) <= 1e-10
    usage = [event.get("usage", {}) for event in result["events"]]
    return {"status": result["status"], "model": result.get("model"),
        "fresh_confirmed": confirmed, "identified": identified,
        "selected_identified_candidate": identified and result.get("model") == survivors[0],
        "confirmed_and_correct": confirmed and correct, "false_confirmation": confirmed and not correct,
        "terminal": terminal, "costs": result["costs"],
        "response_plans": sum(item.get("operator") == "response_disagreement" or
            (item.get("operator") == "research_choice" and item.get("experiment") is not None
             and (item.get("recommendation") or {}).get("operator") == "response_disagreement")
            for item in operators),
        "research_choices": sum(item.get("operator") == "research_choice" for item in operators),
        "legacy_research_calls": sum(item.get("operator") == "legacy_research" for item in operators),
        "measured_response_experiments": len(assessments),
        "method_events": [item["method_feedback"] for item in assessments if item.get("method_feedback")],
        "cached_input_tokens": sum(item.get("cached_input_tokens") or 0 for item in usage),
        "input_tokens": sum(item.get("input_tokens") or 0 for item in usage)}


def historical_methods(path):
    run = json.loads(Path(path).read_text())
    books = {stream["case"]: stream["knowledge"]["program_experiment_methods"]
             for stream in run["streams"]}
    costs = add_costs(task["costs"] for stream in run["streams"] for task in stream["tasks"])
    return books, costs


def run_study(output, *, provider_factory, auth_mode=None, historical_run=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("use a new output directory")
    plan = protocol()
    plan["provider"]["auth_mode"] = auth_mode
    books = None
    if historical_run is not None:
        books, historical_costs = historical_methods(historical_run)
        plan.update(study="historical_response_methods_same_llm_development",
            historical_run=str(Path(historical_run).resolve()), historical_training_costs=historical_costs,
            intervention="Only the frozen historical experiment-method JSON book differs. "
                "Both targets receive an empty lesson and the same current paid calibration, "
                "candidates, tools and limits. No historical concepts or fitted instruments transfer.",
            transfer="Extract the two full method books from the existing measured v0.76 run. "
                "Retain its counterexamples and conditions. Target feedback never reaches another target.")
        (output / "historical_run.json").symlink_to(Path(historical_run).resolve())
    write_json(output / "protocol.json", plan)
    for name in ("scripts/run_temporal_legacy_method_llm.py",
                 "src/hegel_machine/cognition/checkpoint_storage.py",
                 "src/hegel_machine/cognition/temporal_operator_agent.py",
                 "src/hegel_machine/cognition/temporal_response_experiments.py",
                 "src/hegel_machine/cognition/temporal_method_lab.py",
                 "src/hegel_machine/cognition/competing_object_experiments.py",
                 "src/hegel_machine/cognition/program_experiment_methods.py",
                 "src/hegel_machine/cognition/temporal_research_choice.py",
                 "src/hegel_machine/cognition/action_policy.py",
                 "src/hegel_machine/cognition/legacy_research_bridge.py",
                 "src/hegel_machine/cognition/historical_capabilities.py",
                 "src/hegel_machine/cognition/historical_capabilities.json"):
        target = output / "source_at_run" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    config = InquiryLLMConfig(**plan["limits"])

    def execute(case, role, task, book=None, lesson=""):
        path = output / case / role
        agent, lab = prepare_episode(path, task, config, provider_factory(path), book, lesson)
        setup = deepcopy(agent.costs)
        write_json(path / "preparation.json", agent.result("prepared"))
        def checkpoint(result):
            write_json(path / "active.json", result)
            print(json.dumps({"case": case, "role": role, "status": result["status"],
                "calls": result["costs"]["model_calls"], "last_action": result["events"][-1]["action"]}), flush=True)
        result = agent.run(lab, on_checkpoint=checkpoint)
        result["setup_costs"] = setup
        result["lesson"] = agent.lesson
        result["experiment_methods"] = agent._instrument_methods().to_dict()
        write_json(path / "training.json", result)
        return case, role, task, result

    with ThreadPoolExecutor(max_workers=2) as pool:
        sources = (list(pool.map(lambda item: execute(item["case"], "source", item["source"]), plan["cases"]))
                   if books is None else [])
        pending = []
        for index, item in enumerate(plan["cases"]):
            book = books[item["case"]] if books is not None else sources[index][3]["experiment_methods"]
            lesson = "" if books is not None else sources[index][3]["lesson"]
            (output / item["case"]).mkdir(parents=True, exist_ok=True)
            write_json(output / item["case"] / "frozen_methods.json", book)
            if not book:
                continue
            for condition in plan["conditions"]:
                pending.append((item["case"], condition, item["target"],
                                deepcopy(book) if condition == "retained" else None, lesson))
        targets = list(pool.map(lambda args: execute(*args), pending))

    episodes = []
    # No source or target terminal outcomes are available while any agent learns.
    for case, role, task, result in sources + targets:
        lab = LegacyMethodInquiryLab(**task)
        rows = lab.evaluation_rows(length=128, seed=task["seed"] + 1000003)
        terminal = (evaluate_python_model(result["workspace"], result["model"], rows,
                    timeout_seconds=config.execution_seconds) if result.get("model") else {"status": "no_model"})
        terminal.pop("predictions", None)
        terminal.update(rows=len(rows), nonzero_targets=sum(row.target != 0 for row in rows),
                        negative_targets=sum(row.target < 0 for row in rows))
        summary = {"case": case, "role": role, **episode_summary(result, terminal)}
        write_json(output / case / role / "evaluation.json", summary)
        episodes.append(summary)
    summary = {"study": plan["study"], "status": "completed", "episodes": episodes,
        "registered_episodes": 6 if books is None else 4, "attempted_episodes": len(episodes),
        "confirmed_and_correct": sum(item["confirmed_and_correct"] for item in episodes),
        "false_confirmations": sum(item["false_confirmation"] for item in episodes),
        "costs": add_costs(item["costs"] for item in episodes),
        "cached_input_tokens": sum(item["cached_input_tokens"] for item in episodes),
        "input_tokens": sum(item["input_tokens"] for item in episodes)}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-run", type=Path,
                        help="Use measured method books from an existing temporal-method run")
    args = parser.parse_args()
    auth = verify_codex_subscription()
    print(json.dumps(run_study(args.output, auth_mode=auth, historical_run=args.historical_run, provider_factory=lambda path:
        CodexCodeProvider(path, model="gpt-6-luna", reasoning_effort="medium", timeout_seconds=180)), indent=2))


if __name__ == "__main__":
    main()
