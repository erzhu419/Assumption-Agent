"""Practice world research programs with the existing source retention/publisher."""

from copy import deepcopy
import json
from pathlib import Path
from time import monotonic

from .code_provider_context import _compact_practice_feedback
from .discoveryworld_agent import DiscoveryWorldAgent
from .discoveryworld_lab import DiscoveryWorldLab
from .online_inquiry_llm import InquiryLLMConfig
from .python_method_learning import PythonArchiveMethodLearner
from .temporal_method_learning import METHOD_PATH


def compact_author_events(events):
    """Keep practice outcomes in the prompt; leave full executions in trial files."""
    def preview(value, recent):
        if isinstance(value, list):
            return [preview(item, recent) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: preview(item, recent) for key, item in value.items()}
        if isinstance(value.get("steps"), list) and value.get("directory"):
            result["step_count"] = len(value["steps"])
            result["steps"] = result["steps"][-8:] if recent else []
            result["full_trace_file"] = str(Path(value["directory"]) / "result.json")
        return result

    return [preview(_compact_practice_feedback(event), index == len(events) - 1)
            for index, event in enumerate(events)]


class _SourceInterface:
    def __init__(self, spec):
        self.spec = spec

    def public_spec(self):
        return deepcopy(self.spec)


class WorldMethodLearner(PythonArchiveMethodLearner):
    """Keep the old practice history, incumbent comparison and publication flow."""

    def __init__(self, provider, workspace, *, knowledge, task, records, prefix_steps,
                 public_spec, limits, calls, runtime, provider_factory, max_steps=64,
                 laboratory_factory=None, additional_tasks=()):
        self.world_task = deepcopy(task)
        self.source_records = deepcopy(records)
        self.world_limits = deepcopy(limits)
        self.limits = {**limits, "max_model_calls": calls}
        self.started = monotonic()
        self.provider_factory = provider_factory
        self.max_steps = max_steps
        self.world_laboratory_factory = laboratory_factory
        self.world_runtime = runtime
        cases = {"source_initial": {"prefix_steps": 0},
                 "source_continuation": {"prefix_steps": prefix_steps}}
        for additional in additional_tasks:
            cases[f"{additional['id']}_initial"] = {
                "prefix_steps": 0, "task": deepcopy(additional)}
        if not 0 < prefix_steps < len(records):
            raise ValueError("Source practice needs an unfinished public-action prefix")
        config = InquiryLLMConfig(observation_budget=limits["environment_steps"] * calls,
            max_model_calls=calls, token_budget=None,
            execution_seconds=limits["python_seconds_per_task"], compute_budget=10**9)
        super().__init__(provider, workspace, knowledge=knowledge, cases=cases,
            lab_factory=lambda _: _SourceInterface(public_spec), config=config, prepare_trial=None)
        self.costs.update(environment_steps=0, input_tokens=0, output_tokens=0,
                          cached_input_tokens=0, usage_missing_calls=0,
                          wall_time_seconds=0., research_compute_units=0)

    _model_decision = DiscoveryWorldAgent._model_decision

    def _prior_selection_error(self, coverage):
        # This world interface has no archived predictor or numerical submission.
        # The unchanged publisher still requires current-source support on every case.
        return None

    def _execute(self, action, arguments, laboratory=None):
        if "notes" in arguments:
            (self.workspace / "research_notes.md").write_text(arguments["notes"])
        before = (self.workspace / METHOD_PATH).read_text()
        if action == "run" and arguments.get("operator") == "method":
            if arguments.get("path") != METHOD_PATH:
                raise ValueError("Register the self-contained shared/consolidated_method.py")
            # Registration requests the existing paid completion/retention/publish flow.
            return super()._execute("finish", {"path": METHOD_PATH,
                "lesson": arguments.get("applicability", "")}, laboratory)
        feedback = super()._execute(action, arguments, laboratory)
        if action in {"write", "run"} and (self.workspace / METHOD_PATH).read_text() != before:
            missing = self._current_source_coverage()["missing_cases"]
            if missing:
                outcome = self._practice(next(case for case in self.cases if case in missing))
                feedback = {**feedback, "edit_practice": _compact_practice_feedback(outcome),
                            "current_source_coverage": self._current_source_coverage()}
        return feedback

    def _practice(self, case, *, provider_factory=None):
        if case not in self.cases:
            raise ValueError("choose " + " or ".join(self.cases))
        remaining_calls = self.config.max_model_calls - self.costs["model_calls"]
        remaining_seconds = self.config.execution_seconds - self.costs["python_execution_seconds"]
        if remaining_seconds <= 0:
            return {"error": "remaining Python time cannot execute a world practice"}
        if self.costs["observation_cost"] >= self.config.observation_budget:
            return {"error": "remaining observation account cannot execute a world practice"}
        source = (self.workspace / METHOD_PATH).read_text()
        directory = f"trials/trial_{len(self.practice_trials):03d}"
        workspace = self.workspace / directory
        workspace.mkdir(parents=True)
        limits = {**self.world_limits, "max_model_calls": max(0, remaining_calls),
                  "python_seconds_per_task": remaining_seconds}
        factory = self.provider_factory if provider_factory is None else provider_factory
        agent = DiscoveryWorldAgent(factory(workspace), workspace,
                                    arm="hegel", limits=limits)
        prefix_steps = self.cases[case]["prefix_steps"]
        replay_steps = 0
        started = monotonic()
        lab = (self.world_laboratory_factory(workspace) if self.world_laboratory_factory else
               DiscoveryWorldLab(self.cases[case].get("task", self.world_task),
                                 self.world_runtime, workspace))
        with lab:
            agent.lab = lab
            # Replay only published actions, keeping the actual replay observations.
            for record in self.source_records[:prefix_steps]:
                observed = lab.act(record["action"])
                expected = (record["feedback"].get("success"),
                            record["observation"].get("lastActionMessage", ""))
                actual = (observed["feedback"].get("success"),
                          observed["observation"].get("lastActionMessage", ""))
                if actual != expected:
                    raise ValueError(f"Public source replay differs at record {record['index']}")
                agent.observations.append({"index": len(agent.observations), **observed})
                replay_steps += observed.get("step_cost", 1)
            if lab.score()["completed"]:
                raise ValueError("Source practice must start before world completion")
            agent._execute("write", {"path": METHOD_PATH, "code": source}, lab)
            for path, text in self.knowledge.get("files", {}).items():
                if path != METHOD_PATH:
                    agent._execute("write", {"path": path, "code": text}, lab)
            description = "\n".join(row.get("description", "")
                for row in lab.public_observation().get("taskProgress", []))
            agent._question({"id": "task", "question": description})
            agent._save_state()
            execution = agent._run_procedure(lab, METHOD_PATH, max_steps=self.max_steps,
                query_arguments={"procedure_start": len(agent.observations), "parameters": {}})
            raw = agent.result()
            changed = (workspace / METHOD_PATH).read_text() != source
            assisted = any(q.get("origin") == "host_procedure_failure" for q in agent.questions)
            success = raw["score"]["completedSuccessfully"] and not changed and not assisted
            public_progress = deepcopy(lab.public_observation().get("taskProgress", []))
            result = {**raw, "status": "completed" if success else (
                "entry_source_revised" if changed else "repair_assisted" if assisted
                else execution["status"]),
                "model": METHOD_PATH if success else None,
                "fresh_confirmation": {"passed": success, "mse": None,
                    "rows": len(agent.observations) - prefix_steps,
                    "kind": "new_world_execution_on_source_practice"},
                "execution": execution, "public_task_progress": public_progress,
                "replayed_environment_steps": replay_steps,
                "new_environment_steps": lab.environment_steps - replay_steps,
                "entry_source_changed": changed,
                "automatic_repair_used": assisted,
                "executed_source": (workspace / METHOD_PATH).read_text(),
                "paid_observations": deepcopy(agent.observations)}
            result["costs"].update(observation_cost=lab.environment_steps,
                compute_cost=raw["costs"]["research_compute_units"],
                environment_steps=lab.environment_steps,
                wall_time_seconds=monotonic() - started)
        revised_path = Path(directory) / "shared/revised_method.py"
        if changed:
            (self.workspace / revised_path).write_text(result["executed_source"])
            # Candidate source_path and restore_method must identify the same draft.
            (workspace / METHOD_PATH).write_text(source)
        # The old recorder owns the four ordinary practice costs; merge only the
        # remaining child-model/account fields here, exactly once.
        feedback = self._record_practice(case, directory, source, agent, result, False)
        for key in ("model_calls", "reported_tokens", "input_tokens", "output_tokens",
                    "cached_input_tokens", "usage_missing_calls", "environment_steps",
                    "wall_time_seconds", "research_compute_units"):
            self.costs[key] += result["costs"].get(key, 0)
        feedback.update(public_task_progress=public_progress,
                        new_environment_steps=result["new_environment_steps"],
                        replayed_environment_steps=replay_steps, entry_source_changed=changed,
                        automatic_repair_used=assisted,
                        revised_source_path=str(revised_path) if changed else None,
                        repair_trace_file=str(Path(directory) / "result.json"))
        if changed and raw["score"]["completedSuccessfully"] and provider_factory is None:
            # A successful interactive repair is a candidate. Reuse the same
            # practice with no model before retaining it as the author's draft.
            candidate = result["executed_source"]
            self.files[METHOD_PATH] = candidate
            (self.workspace / METHOD_PATH).write_text(candidate)
            confirmation = self._practice(case, provider_factory=lambda _: None)
            retained = (confirmation.get("fresh_confirmation") or {}).get("passed") is True
            if not retained:
                self.files[METHOD_PATH] = source
                (self.workspace / METHOD_PATH).write_text(source)
            feedback["repaired_candidate_confirmation"] = {
                "retained": retained, "practice": _compact_practice_feedback(confirmation)}
            feedback["current_source_coverage"] = self._current_source_coverage()
        return feedback

    def public_practice_archive(self):
        source = (self.workspace / METHOD_PATH).read_text()
        trials = []
        for trial in self._latest_source_practices(source).values():
            result = json.loads((self.workspace / trial["directory"] / "result.json").read_text())
            trials.append({"case": trial["case"], "status": trial["status"],
                "public_task_progress": result["public_task_progress"],
                "observations": [DiscoveryWorldAgent._action_observation(row)
                                 for row in result["paid_observations"]],
                "steps": result["method_results"], "costs": result["costs"]})
        return {"origin": "Actual source-world method executions, separate from target observations",
                "trials": trials}
