#!/usr/bin/env python3
"""Review paid source evidence, consolidate with source practice, then transfer."""

import argparse
import ast
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent, INTERFACE, INQUIRY_INTERFACE
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from hegel_machine.cognition.world_method_learning import WorldMethodLearner, run_world_method_author
from hegel_machine.cognition.world_method_transfer import install_method_bundle
from hegel_machine.cognition.world_prerequisite_research import public_reference_history
from scripts.prepare_discoveryworld_transfer_method import AUTHOR_TASK
from scripts.run_discoveryworld_method_repair import restore_repair
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_response_measurement_recovery import AliasedSignalLab


CASES = {
    "parameter_low": {"intercept": 40.2, "gain": 12., "power": 2, "sensor": "informative"},
    "parameter_high": {"intercept": 220.4, "gain": 7.5, "power": 2, "sensor": "informative"},
    "uninformative_sensor": {"intercept": 40.2, "gain": 12., "power": 2, "sensor": "density"},
    "changed_mechanism": {"intercept": 40.2, "gain": 1., "power": 3, "sensor": "informative"},
}
REVIEW_TASK = """
OPERATOR-REQUESTED SOURCE ANALYSIS REVIEW: review the unresolved source conclusion
for reference_extension_5 against the existing paid evidence. Inspect its existing
analysis and notes; repair and evaluate the conclusion if warranted. No new world
experiments in this review. You may inspect/calculate, write an analysis, evaluate,
or finish. Decide the conclusion yourself. Method consolidation is a later stage.
"""
FROZEN_TASK = """
Every source practice here executes without an LLM. Failures return to you for
revision. Consolidate the source experience into the existing working program;
source-world practice is not held-out transfer. No new target is available here.
"""


class TransferSignalLab(AliasedSignalLab):
    def __init__(self, case):
        super().__init__("informative")
        config = CASES[case]
        if "signal_values" in config:
            self.signal = {int(n): float(value) for n, value in config["signal_values"].items()}
        self.required = {n: config["intercept"] + config["gain"] * value ** config["power"]
                         for n, value in self.signal.items()}
        self.frequencies = {n: self.required[n] if n <= 3 else 500 for n in self.signal}
        if config["sensor"] == "density":
            self.signal = dict(self.density)

    def act(self, action):
        result = super().act(action)
        # Reuse the existing ImmediateCompletionLab semantics: activation ends
        # the task even while the control dialog is open, after range clamping.
        self.completed = self.success = all(self.active(n) for n in (4, 5))
        result["observation"] = self.public_observation()
        return result


def review_source(source, output, provider_factory, calls=6, reuse_result=None,
                  laboratory_factory=AliasedSignalLab):
    """Keep the source's account and observations; the operator requests review."""
    saved = json.loads((source / "result.json").read_text())
    previous = (json.loads(Path(reuse_result).read_text()) if reuse_result
                else deepcopy(saved.get("agent", saved)))
    previous["episode_status"] = previous["status"]
    previous_workspace = Path(reuse_result).parent / "agent" if reuse_result else source / "agent"
    if reuse_result:
        calls = 0
    limits = deepcopy(json.loads((source / "protocol.json").read_text())["limits"])
    limits.update(max_model_calls=previous["costs"]["model_calls"] + calls,
                  task_wall_seconds=previous["costs"]["wall_time_seconds"] + 1800,
                  python_seconds_per_task=previous["costs"]["python_execution_seconds"] + 120)
    agent = DiscoveryWorldAgent(None if reuse_result else provider_factory(output / "agent"), output / "agent",
                                arm="hegel", limits=limits)
    lab = laboratory_factory()
    agent.lab = lab
    replay = restore_repair(agent, lab, previous, previous_workspace)
    before = deepcopy(agent.costs)
    first_event = len(agent.events)

    def save(status):
        agent._save_state()
        write_json(output / "result.json", {**agent.result(), "review_status": status,
            "trigger": "operator_requested_source_analysis_review", "replay": replay,
            "reused_review_result": str(Path(reuse_result).resolve()) if reuse_result else None,
            "costs_before": before, "new_calls": agent.costs["model_calls"] - before["model_calls"],
            "new_actions": lab.environment_steps - before["environment_steps"],
            "review_events": deepcopy(agent.events[first_event:])})

    status = "reviewing"
    if reuse_result:
        save("reused_paid_review")
        return agent, lab
    save(status)
    while agent.costs["model_calls"] < limits["max_model_calls"]:
        question = next(q for q in agent.questions if q["id"] == "reference_extension_5")
        if question["status"] in ("supported", "refuted"):
            status = "conclusion_revised"
            break
        request = agent._request()
        request["task_interface"] += REVIEW_TASK
        decision = agent._model_decision(request)
        action, arguments = decision["action"], decision.get("arguments", {})
        allowed = (action in ("evaluate", "finish") or
            action == "write" and arguments.get("path") != METHOD_PATH or
            action == "run" and not arguments.get("operator"))
        try:
            if action == "provider_error":
                feedback = {"error": decision.get("error")}
            else:
                feedback = agent._execute(action, arguments, lab) if allowed else {
                    "error": "This paid-evidence review allows analysis, evaluation and finish; no world execution."}
        except (ValueError, KeyError, TypeError, OSError) as error:
            feedback = {"error": f"{type(error).__name__}: {error}"}
        agent.events.append({"action": action, "arguments": deepcopy(arguments),
                             "feedback": feedback, "usage": decision.get("usage", {})})
        save(status)
        if action == "finish":
            status = "review_finished"
            break
        if action == "provider_error" and not (isinstance(decision.get("error"), dict)
                                               and decision["error"].get("decode_error")):
            status = "provider_failed"
            break
    else:
        status = "review_call_limit"
    if next(q for q in agent.questions if q["id"] == "reference_extension_5")["status"] in ("supported", "refuted"):
        status = "conclusion_revised"
    save(status)
    return agent, lab


def source_knowledge(source, reviewed, prefix_steps):
    original = json.loads((source / "result.json").read_text())["agent"]
    observations = reviewed.observations
    archive = {"origin": "Public source experience; not current practice observations",
        "review_origin": "operator_requested_source_analysis_review",
        "public_spec": {"capabilities": reviewed.lab.capabilities(),
                        "current_observation": observations[-1]["observation"]},
        "original_questions": original["research"]["questions"], "questions": deepcopy(reviewed.questions),
        "original_inquiry_notes": {k: v.get("notes", "") for k, v in original["research"]["llm_inquiries"].items()},
        "inquiry_notes": {k: v.get("notes", "") for k, v in reviewed.llm_inquiries.items()},
        "notes": (reviewed.workspace / "research_notes.md").read_text()
            if (reviewed.workspace / "research_notes.md").exists() else "",
        "measurement_history": [DiscoveryWorldAgent._action_observation(r) for r in observations
                                if r["action"].get("action") in ("USE", "READ")],
        "reference_history": public_reference_history(observations),
        "continuation_observations": [DiscoveryWorldAgent._action_observation(r) for r in observations[prefix_steps:]],
        "events": deepcopy(reviewed.events[len(original["events"]):])}
    files = {METHOD_PATH: (reviewed.workspace / METHOD_PATH).read_text(),
        "shared/source_episode.json": json.dumps(archive, ensure_ascii=False),
        "shared/source_original_method.py": (source / "agent" / METHOD_PATH).read_text()}
    from hegel_machine.cognition.world_response_transfer import (
        KNOWLEDGE_PATH, combine_response_knowledge, export_response_knowledge)

    inherited_path = reviewed.workspace / KNOWLEDGE_PATH
    inherited = json.loads(inherited_path.read_text()) if inherited_path.exists() else None
    frozen = None
    state_path = reviewed.workspace / 'shared/reference_recalibration.json'
    if state_path.exists():
        from hegel_machine.cognition.world_reference_recalibration import refresh_response_evidence

        declarations = next((ast.literal_eval(node.value) for node in ast.parse(files[METHOD_PATH]).body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'ESTIMATORS'
                for target in node.targets)), [])
        state = json.loads(state_path.read_text())
        if inherited is None:
            inherited = state.get('frozen_response_knowledge')
        refresh_response_evidence(state, observations, reviewed.lab.public_observation(), declarations)
        frozen = export_response_knowledge(state, declarations, str(source))
    retained = combine_response_knowledge(inherited, frozen)
    if retained and retained['models']:
        files[KNOWLEDGE_PATH] = json.dumps(retained)
    for phase, workspace in (("original", source / "agent"), ("reviewed", reviewed.workspace)):
        for path in (workspace / "shared").glob("analysis_*.py"):
            files[f"shared/{phase}_{path.name}"] = path.read_text()
    return {"files": files}, archive


def previous_author_experience(path):
    """Retain failed drafts and paid feedback as experience, never as support."""
    previous = json.loads(Path(path).read_text())
    workspace = Path(previous["workspace"])
    draft = (workspace / METHOD_PATH).read_text()
    candidates = next((ast.literal_eval(node.value) for node in ast.parse(draft).body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ESTIMATORS"
                                               for t in node.targets)), [])
    experience = {"origin": "Previous source author; failed publication is not current-source support",
        "status": previous["status"], "costs": previous["costs"], "candidates": candidates,
        "notes": (workspace / "research_notes.md").read_text()
            if (workspace / "research_notes.md").exists() else "",
        "draft_path": "shared/previous_author_method.py", "practices": [], "events": previous["events"]}
    for trial in previous["practice_trials"]:
        result = json.loads((workspace / trial["directory"] / "result.json").read_text())
        experience["practices"].append({"case": trial["case"], "status": result["status"],
            "costs": result["costs"], "execution": result["execution"],
            "observations": [DiscoveryWorldAgent._action_observation(r) for r in result["paid_observations"]]})
    return {"shared/previous_author_method.py": draft,
            "shared/previous_author_experience.json": json.dumps(experience, ensure_ascii=False)}, experience


def run_transfer_case(case, arm, package, output, limits, *, provider_factory=None,
                      on_checkpoint=None):
    """Install an existing package; optionally let its own inquiry call a model."""
    directory = Path(output) / case / arm
    lab = TransferSignalLab(case)
    provider = provider_factory(directory / "agent") if provider_factory else None
    agent = DiscoveryWorldAgent(provider, directory / "agent", arm="hegel",
                                limits=limits if provider is not None else {**limits, "max_model_calls": 0})
    agent.lab = lab
    agent._on_checkpoint = on_checkpoint
    install_method_bundle(agent, package, lab)
    agent._question({"id": "task", "question": "Calibrate the two targets"})
    execution = agent._execute("run", {"operator": "procedure",
        "method_id": package["methods"][0]["id"], "max_steps": 256}, lab)
    unchanged = (agent.workspace / METHOD_PATH).read_text() == package["files"][METHOD_PATH]
    result = agent.result()
    result["costs"].update(observation_cost=lab.environment_steps,
        compute_cost=result["costs"]["research_compute_units"])
    write_json(directory / "result.json", {**result, "execution": execution,
        "source_unchanged": unchanged, "observations": deepcopy(agent.observations)})
    return {"case": case, "arm": arm, "success": result["score"]["completedSuccessfully"],
        "status": execution["status"], "source_unchanged": unchanged,
        "execution_kind": ("live_research" if provider is not None else
                           "frozen_source" if unchanged else "zero_llm_target_adaptation"),
        "costs": result["costs"]}


def transfer(bundle, old_source, output, limits):
    rows = []
    old = {"files": {METHOD_PATH: old_source}, "methods": [{"id": "source_program",
        "path": METHOD_PATH, "parameters": {}, "requires": [], "max_steps": 256}]}
    for case in CASES:
        for arm, package in (("previous_source", old), ("consolidated", bundle)):
            rows.append(run_transfer_case(case, arm, package, output, limits))
            write_json(output / "brief.json", rows)
    return rows


def run(source, output, *, provider_factory, calls=24, review_calls=6, auth_mode="scripted_test",
        initial_bundle=None, review_result=None, previous_author=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads((source / "result.json").read_text())
    if not original["agent"]["score"]["completedSuccessfully"]:
        raise ValueError("Consolidation requires an actually successful source")
    prefix_steps = len(json.loads((source / "prefix.json").read_text())["observations"])
    limits = deepcopy(json.loads((source / "protocol.json").read_text())["limits"])
    limits.update(max_model_calls=calls, task_wall_seconds=3600, python_seconds_per_task=240,
                  provider_request_timeout_seconds=180)
    protocol = {"source": str(source), "model": {"requested_alias": "gpt-6-luna", "reasoning_effort": "medium"},
        "auth_mode": auth_mode, "source_costs": original["agent"]["costs"], "review_calls": 0 if review_result else review_calls,
        "author_calls": calls, "limits": limits, "prefix_steps": prefix_steps,
        "source_practices": ["source_initial", "source_continuation"], "target_conditions": CASES,
        "initial_bundle": str(Path(initial_bundle).resolve()) if initial_bundle else None,
        "reused_review_result": str(Path(review_result).resolve()) if review_result else None,
        "previous_author_result": str(Path(previous_author).resolve()) if previous_author else None,
        "scope": "Operator-requested paid-source review; source-program practice and consolidation; zero-LLM "
                 "paired transfer only after publication. Negative conditions do not prespecify failure. "
                 "Target source edits are reported as adaptation; target parameters are not given to the author."}
    write_json(output / "protocol.json", protocol)
    reviewed, _ = review_source(source, output / "source_review", provider_factory, review_calls, review_result)
    knowledge, archive = source_knowledge(source, reviewed, prefix_steps)
    if initial_bundle:
        knowledge["files"][METHOD_PATH] = json.loads(Path(initial_bundle).read_text())["files"][METHOD_PATH]
    initial_source = knowledge["files"][METHOD_PATH]
    knowledge["files"]["shared/phase_initial_method.py"] = initial_source
    prior_experience = None
    if previous_author:
        files, prior_experience = previous_author_experience(previous_author)
        knowledge["files"].update(files)
    reviewed_costs = json.loads((Path(review_result) if review_result else output / "source_review/result.json").read_text())["costs"]
    protocol["prior_work"] = {"source": protocol["source_costs"],
        "source_review": {key: value - protocol["source_costs"].get(key, 0) for key, value in reviewed_costs.items()},
        "previous_author": prior_experience["costs"] if prior_experience else None,
        "accounting": "Source, paid review and earlier author are separate prior costs. Current author costs start at zero; "
            "only entry code seeds this phase, all source practices and support are acquired anew."}
    protocol["transfer_baseline"] = "This phase's starting source, including the same executor corrections as the author seed."
    write_json(output / "protocol.json", protocol)
    workspace = output / "author"
    learner = WorldMethodLearner(provider_factory(workspace), workspace, knowledge=knowledge,
        task={"id": "measurement_source"}, records=reviewed.observations, prefix_steps=prefix_steps,
        public_spec=archive["public_spec"], limits=limits, calls=calls, runtime="existing_fixture",
        provider_factory=lambda _: None, laboratory_factory=lambda _: nullcontext(AliasedSignalLab()), max_steps=256)

    def checkpoint(status):
        write_json(output / "result.json", {**learner.result(status), "source": str(source),
            "source_costs": protocol["source_costs"], "prior_work": protocol["prior_work"],
            "review_result": str(output / "source_review/result.json")})
        print(json.dumps({"phase": "source_author", "status": status, "calls": learner.costs["model_calls"]}), flush=True)

    for case in learner.cases:
        feedback = learner._practice(case)
        learner.events.append({"action": "run", "arguments": {"operator": "practice", "case": case},
                               "feedback": feedback, "origin": "initial_source_practice"})
    notes = "\n\n".join(["Source notebook:\n" + archive["notes"], *[
        f"Original source inquiry {key}:\n{value}" for key, value in archive["original_inquiry_notes"].items()], *[
        f"Reviewed source inquiry {key}:\n{value}" for key, value in archive["inquiry_notes"].items()]])
    experiences = [{"file": "shared/source_episode.json", "notes": notes,
        "sections": {key: len(archive[key]) for key in ("measurement_history", "reference_history",
            "continuation_observations", "events", "questions")}}]
    if prior_experience:
        experiences.append({"file": "shared/previous_author_experience.json", "notes": prior_experience["notes"],
            "sections": {key: len(prior_experience[key]) for key in ("candidates", "practices", "events")}})
    status = run_world_method_author(learner, task_interface=INTERFACE + INQUIRY_INTERFACE + AUTHOR_TASK + FROZEN_TASK,
        source_experiences=experiences, on_checkpoint=checkpoint)
    if not learner.published_source:
        return {"status": status, "transfer": "not_run_unpublished_draft"}
    practice = learner.public_practice_archive()
    bundle = {"files": {METHOD_PATH: learner.published_source,
        "shared/source_practice.json": json.dumps(practice, ensure_ascii=False)}, "methods": [{
        "id": "practiced_measurement_research", "path": METHOD_PATH, "parameters": {}, "requires": [],
        "max_steps": 256, "evidence_file": "shared/source_practice.json", "applicability": learner.lesson}],
        "source": str(source), "consolidation_costs": deepcopy(learner.costs)}
    write_json(output / "method_bundle.json", bundle)
    rows = transfer(bundle, initial_source, output / "transfer", limits)
    return {"status": status, "transfer": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("codex_cli",))
    parser.add_argument("--calls", type=int, default=24)
    parser.add_argument("--review-calls", type=int, default=6)
    parser.add_argument("--initial-bundle", type=Path, help="Seed entry source only; no previous practice support")
    parser.add_argument("--review-result", type=Path, help="Reuse the existing paid source review without a new review call")
    parser.add_argument("--previous-author", type=Path, help="Earlier failed author result retained as experience and prior costs")
    args = parser.parse_args()
    auth = verify_codex_subscription()
    factory = lambda path: CodexCodeProvider(path, model="gpt-6-luna", reasoning_effort="medium", timeout_seconds=180)
    print(json.dumps(run(args.source, args.output, provider_factory=factory, calls=args.calls,
                        review_calls=args.review_calls, auth_mode=auth, initial_bundle=args.initial_bundle,
                        review_result=args.review_result, previous_author=args.previous_author), indent=2))
