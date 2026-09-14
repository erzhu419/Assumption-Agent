"""Continuing ordinary Python control on the common unknown-task interface."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import shutil

from .code_agent import CodeAgent, CodeAgentConfig, CodexCodeProvider, evaluate_python_model
from .models import Probe
from .unified_session import laboratory_for


class UnifiedCodeSession:
    """Let the existing LLM tool loop learn and carry its own Python knowledge.

    Task descriptions used to construct the laboratory stay outside the provider
    context. Each task has the frozen per-task resource caps, and final evaluation
    starts only after the learning sequence has stopped.
    """

    def __init__(self, workspace, *, tasks=None, config=None, state=None,
                 provider_factory=None, lab_factory=None):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.provider_factory = provider_factory or (lambda path: CodexCodeProvider(path))
        self.lab_factory = lab_factory or laboratory_for
        saved = state or {}
        self.config = config or CodeAgentConfig(**saved.get("config", {}))
        if isinstance(self.config, dict):
            self.config = CodeAgentConfig(**self.config)
        if saved:
            self.state = deepcopy(saved)
            if tasks is not None and list(tasks) != self.state["tasks"]:
                raise ValueError("resume the saved common-task sequence")
            if asdict(self.config) != self.state["config"]:
                raise ValueError("resume uses the frozen per-task resource limits")
        else:
            self.state = {"agent": "ordinary_python_code_agent", "status": "ready",
                "tasks": deepcopy(list(tasks or [])), "config": asdict(self.config),
                "results": [], "active": None, "knowledge": {"files": {}, "lessons": []},
                "evaluation": None, "costs": {}}
        self._costs()

    def _costs(self):
        rows = [result["costs"] for result in self.state["results"]]
        if self.state["active"]:
            rows.append(self.state["active"]["agent_state"]["costs"])
        names = ("observation_cost", "measurement_cost", "compute_cost",
                 "python_execution_seconds", "model_calls", "reported_tokens")
        self.state["costs"] = {name: (None if any(row.get(name) is None for row in rows)
            else sum(row[name] for row in rows)) for name in names}

    @staticmethod
    def _restore_acquisitions(laboratory, events):
        for event in events:
            if event.get("action") == "query" and event.get("feedback", {}).get("observation_cost"):
                arguments = event["arguments"]
                laboratory.acquire(Probe(f"code_query_{event['step']}", tuple(arguments["sequence"])),
                                   tuple(arguments.get("channels", ())))

    def run(self, max_tasks=None, on_checkpoint=None):
        visited = 0
        while len(self.state["results"]) < len(self.state["tasks"]) and (
                max_tasks is None or visited < max_tasks):
            index = len(self.state["results"])
            item = self.state["tasks"][index]
            active = self.state["active"]
            inherited = deepcopy(self.state["knowledge"])
            inherited.update(task_position=index, tasks_remaining=len(self.state["tasks"])-index)
            task_workspace = self.workspace / f"task_{index:03d}"
            if index and not active:
                previous_shared = Path(self.state["results"][-1]["research_workspace"]) / "shared"
                if previous_shared.is_dir():
                    shutil.copytree(previous_shared, task_workspace / "shared", dirs_exist_ok=True)
            agent = CodeAgent(self.provider_factory(task_workspace), task_workspace, self.config,
                state=active["agent_state"] if active else None, session_knowledge=inherited)

            def checkpoint(agent_state):
                self.state["active"] = {"task_position": index, "agent_state": agent_state}
                self._costs()
                if on_checkpoint:
                    on_checkpoint(self.to_dict())

            with self.lab_factory(item) as laboratory:
                if active:
                    self._restore_acquisitions(laboratory, agent.events)
                result = agent.run(laboratory, on_checkpoint=checkpoint)
            saved = agent.to_dict()
            result.update(task_position=index, inherited_files=sorted(inherited["files"]),
                          inherited_lessons=len(inherited["lessons"]))
            self.state["results"].append(result)
            self.state["active"] = None
            self.state["knowledge"]["files"].update({name: source for name, source in saved["files"].items()
                                                     if name.startswith("shared/")})
            self.state["knowledge"]["shared_artifacts"] = sorted(str(path.relative_to(task_workspace))
                for path in (task_workspace / "shared").rglob("*") if path.is_file()
                and str(path.relative_to(task_workspace)) not in saved["files"]
                and "__pycache__" not in path.parts)
            # Only observed training feedback and agent-authored knowledge move
            # forward. Private laboratory settings and final labels never do.
            self.state["knowledge"]["lessons"].append({"task_position": index,
                "status": result["status"], "lesson": result["lesson"],
                "selected_channels": result["selected_channels"],
                "training_feedback": next((event["feedback"] for event in reversed(result["events"])
                    if event.get("action") in {"finish", "evaluate"}), None),
                "model_source": result["model_source"], "costs": result["costs"]})
            visited += 1
            self._costs()
            self.state["status"] = ("completed" if len(self.state["results"]) == len(self.state["tasks"])
                                    else "paused")
            if result["status"] in {"provider_failed", "token_usage_unavailable"}:
                self.state["status"] = result["status"]
            if on_checkpoint:
                on_checkpoint(self.to_dict())
            if result["status"] in {"provider_failed", "token_usage_unavailable"}:
                break
        return self.to_dict()

    def evaluate(self, *, seed_offset=1000003, length=64, trajectories=4):
        if len(self.state["results"]) != len(self.state["tasks"]):
            raise ValueError("finish the learning sequence before final evaluation")
        if self.state["evaluation"] is not None:
            return deepcopy(self.state["evaluation"])
        scores = []
        for index, (item, result) in enumerate(zip(self.state["tasks"], self.state["results"])):
            with self.lab_factory(item) as laboratory:
                risk_quantile = laboratory.spec.risk_quantile
                if not result["model"]:
                    scores.append({"task_position": index, "status": "missing_model", "mse": None,
                        "loss": None, "loss_kind": "mse" if risk_quantile == .5 else "pinball",
                        "risk_quantile": risk_quantile, "rows": 0,
                        "observation_cost": 0, "measurement_cost": 0, "compute_cost": 0})
                    continue
                rows = laboratory.evaluation_rows(result["selected_channels"],
                    seed=item["seed"]+seed_offset, length=length, trajectories=trajectories)
                rate = sum(channel.cost_per_row for channel in laboratory.channels
                           if channel.name in result["selected_channels"])
            measured = evaluate_python_model(result["workspace"], result["model"], rows,
                timeout_seconds=self.config.execution_seconds, risk_quantile=risk_quantile)
            measured.pop("predictions", None)
            measured.update(task_position=index, rows=len(rows), observation_cost=len(rows),
                            measurement_cost=len(rows) * rate, compute_cost=len(rows),
                            status="completed" if measured["returncode"] == 0 else "model_failed")
            scores.append(measured)
        self.state["evaluation"] = scores
        return deepcopy(scores)

    def to_dict(self):
        return deepcopy(self.state)


def run_code_sequence(tasks, *, workspace, config=None, state=None, provider_factory=None,
                      lab_factory=None, max_tasks=None, on_checkpoint=None):
    """Run or resume learning; callers evaluate after all comparison arms stop."""
    return UnifiedCodeSession(workspace, tasks=tasks, config=config, state=state,
        provider_factory=provider_factory, lab_factory=lab_factory).run(
            max_tasks=max_tasks, on_checkpoint=on_checkpoint)
