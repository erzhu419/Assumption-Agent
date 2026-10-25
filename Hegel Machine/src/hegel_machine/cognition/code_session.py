"""Continuing ordinary Python research and a shared-resource Hegel comparison."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
from time import monotonic

from .code_agent import CodeAgent, CodeAgentConfig, CodexCodeProvider, _copy_model_file, evaluate_python_model
from .evaluation import loss
from .language import Program
from .measurement_lab import make_lab
from .measurement_research import MeasurementResearchAgent
from .memory import KnowledgeLibrary
from .models import Probe, ResearchConfig


_BUDGET_COST = {"observation_budget": "observation_cost", "measurement_budget": "measurement_cost",
               "compute_budget": "compute_cost", "execution_seconds": "python_execution_seconds",
               "max_model_calls": "model_calls", "token_budget": "reported_tokens"}


class CodeResearchSession:
    """Persist agent-written libraries, lessons and the paid task frontier.

    Task checkpoints retain the workspace paths, including arbitrary binary fit
    products. Shared Python and notes are also included in the JSON checkpoint.
    """

    def __init__(self, workspace, *, tasks=None, config=None, totals=None, backend="process",
                 state=None, provider_factory=None):
        saved = state or {}
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.config = config or CodeAgentConfig(**saved.get("config", {}))
        if isinstance(self.config, dict):
            self.config = CodeAgentConfig(**self.config)
        self.backend = saved.get("backend", backend)
        self.provider_factory = provider_factory or (lambda path: CodexCodeProvider(path))
        if saved:
            self.state = deepcopy(saved)
            self.state["config"] = asdict(self.config)
            if totals is not None:
                if any(totals[key] < saved["totals"][key] for key in _BUDGET_COST):
                    raise ValueError("resume may only increase shared resource limits")
                self.state["totals"] = dict(totals)
            if tasks is not None and list(tasks) != saved["tasks"]:
                raise ValueError("resume the saved task sequence; use add_task for new arrivals")
        else:
            tasks = list(tasks or [])
            total = totals or {key: getattr(self.config, key) * max(1, len(tasks)) for key in _BUDGET_COST}
            self.state = {"agent": "continuing_ordinary_python_code_agent", "status": "ready",
                "config": asdict(self.config), "totals": dict(total), "tasks": deepcopy(tasks),
                "backend": backend, "results": [], "active": None,
                "knowledge": {"files": {}, "lessons": []},
                "costs": {name: 0 for name in _BUDGET_COST.values()}}

    def add_task(self, task_index, seed):
        self.state["tasks"].append({"task_index": task_index, "seed": seed})
        self.state["status"] = "ready"

    def _costs(self):
        revisiting = (self.state["active"] or {}).get("task_position")
        rows = [result["costs"] for index, result in enumerate(self.state["results"]) if index != revisiting]
        if self.state["active"]:
            rows.append(self.state["active"]["agent_state"]["costs"])
        self.state["costs"] = {key: (None if any(row[key] is None for row in rows)
                                    else sum(row[key] for row in rows)) for key in _BUDGET_COST.values()}

    def continue_task(self, task_position, *, config=None, on_checkpoint=None):
        """Finish an interrupted or resource-limited task using its paid rows."""
        previous = self.state["results"][task_position]
        if previous["model"]:
            return deepcopy(previous)
        config = config or self.config
        task_workspace = Path(previous["workspace"])
        active = self.state["active"]
        if active and active.get("task_position") == task_position:
            agent_state = active["agent_state"]
        else:
            files = {name: source for name, source in self.state["knowledge"]["files"].items()}
            for event in previous["events"]:
                arguments = event.get("arguments", {})
                if event["action"] == "write":
                    path = task_workspace / arguments["path"]
                    if path.is_file():
                        files[arguments["path"]] = path.read_text()
            agent_state = {"observations": json.loads((task_workspace / "observations.json").read_text()),
                "files": files, "events": previous["events"], "costs": previous["costs"],
                "tokens_available": previous["costs"]["reported_tokens"] is not None,
                "model": None, "selected_channels": previous["selected_channels"], "lesson": "", "reused_files": []}
        other_costs = {name: self.state["costs"][name] - agent_state["costs"][name] for name in _BUDGET_COST.values()}
        config = replace(config, **{key: min(getattr(config, key), self.state["totals"][key]-other_costs[cost])
                                    for key, cost in _BUDGET_COST.items()})
        agent = CodeAgent(self.provider_factory(task_workspace), task_workspace, config,
                          state=agent_state, session_knowledge=self.state["knowledge"])

        def checkpoint(saved):
            self.state["active"] = {"task_position": task_position, "config": asdict(config), "agent_state": saved}
            self._costs()
            if on_checkpoint:
                on_checkpoint(self.to_dict())

        with make_lab(previous["task_index"], previous["seed"], backend=self.backend) as laboratory:
            for event in agent.events:
                if event["action"] == "query" and event.get("feedback", {}).get("observation_cost"):
                    arguments = event["arguments"]
                    laboratory.acquire(Probe(f"code_query_{event['step']}", tuple(arguments["sequence"])),
                                       tuple(arguments.get("channels", ())))
            result = agent.run(laboratory, on_checkpoint=checkpoint)
        result.update(**{key: previous[key] for key in ("task_position", "task_index", "seed", "inherited_files", "inherited_lessons")})
        result["previous_attempt_status"] = previous["status"]
        self.state["results"][task_position] = result
        self.state["active"] = None
        self.state["knowledge"]["files"].update({name: source for name, source in agent.to_dict()["files"].items()
                                                if name.startswith("shared/")})
        self.state["knowledge"]["lessons"][task_position].update(status=result["status"], lesson=result["lesson"],
            model_source=result["model_source"], selected_channels=result["selected_channels"], costs=result["costs"],
            training_feedback=next((event["feedback"] for event in reversed(result["events"])
                                    if event["action"] in {"finish", "evaluate"}), None))
        self._costs()
        self.state["status"] = "completed" if len(self.state["results"]) == len(self.state["tasks"]) else "paused"
        if on_checkpoint:
            on_checkpoint(self.to_dict())
        return deepcopy(result)

    def run(self, max_tasks=None, on_checkpoint=None):
        if max_tasks != 0 and self.state["active"] and "task_position" in self.state["active"]:
            active = self.state["active"]
            self.continue_task(active["task_position"], config=CodeAgentConfig(**active["config"]),
                               on_checkpoint=on_checkpoint)
        completed = 0
        while len(self.state["results"]) < len(self.state["tasks"]) and (max_tasks is None or completed < max_tasks):
            index = len(self.state["results"])
            item = self.state["tasks"][index]
            active = self.state["active"]
            costs = self.state["costs"]
            if costs["reported_tokens"] is None:
                self.state["status"] = "token_usage_unavailable"
                break
            remaining = {limit: self.state["totals"][limit]-costs[cost] for limit, cost in _BUDGET_COST.items()}
            if not active and any(remaining[key] <= 0 for key in ("observation_budget", "compute_budget", "execution_seconds", "max_model_calls", "token_budget")):
                self.state["status"] = "budget_exhausted"
                break
            config = (CodeAgentConfig(**active["config"]) if active else replace(self.config,
                **{key: min(getattr(self.config, key), value) for key, value in remaining.items()}))
            task_workspace = self.workspace / f"task_{index:03d}"
            inherited = deepcopy(self.state["knowledge"])
            inherited.update(task_position=index, tasks_remaining=len(self.state["tasks"])-index,
                             total_resources=self.state["totals"], spent_resources=costs)
            if index and not active:
                previous_shared = Path(self.state["results"][-1]["workspace"]) / "shared"
                if previous_shared.is_dir():
                    shutil.copytree(previous_shared, task_workspace / "shared", dirs_exist_ok=True,
                                    copy_function=_copy_model_file)
            agent = CodeAgent(self.provider_factory(task_workspace), task_workspace, config,
                              state=active["agent_state"] if active else None, session_knowledge=inherited)

            def checkpoint(agent_state):
                self.state["active"] = {"config": asdict(config), "agent_state": agent_state}
                self._costs()
                if on_checkpoint:
                    on_checkpoint(self.to_dict())

            with make_lab(item["task_index"], item["seed"], backend=self.backend) as laboratory:
                if active:
                    # Reconnect the saved simulator trajectory in acquisition
                    # order, as the instrumented research checkpoint does.
                    for event in agent.events:
                        if event["action"] == "query" and event.get("feedback", {}).get("observation_cost"):
                            arguments = event["arguments"]
                            laboratory.acquire(Probe(f"code_query_{event['step']}", tuple(arguments["sequence"])),
                                               tuple(arguments.get("channels", ())))
                result = agent.run(laboratory, on_checkpoint=checkpoint)
            saved = agent.to_dict()
            result.update(**item, task_position=index, inherited_files=sorted(inherited["files"]),
                          inherited_lessons=len(inherited["lessons"]))
            self.state["results"].append(result)
            self.state["active"] = None
            self.state["knowledge"]["files"].update({name: source for name, source in saved["files"].items()
                                                    if name.startswith("shared/")})
            self.state["knowledge"]["lessons"].append({**item, "status": result["status"],
                "lesson": result["lesson"], "selected_channels": result["selected_channels"],
                "training_feedback": next((event["feedback"] for event in reversed(result["events"])
                                            if event["action"] in {"finish", "evaluate"}), None),
                "model_source": result["model_source"], "costs": result["costs"]})
            self._costs()
            completed += 1
            self.state["status"] = "completed" if len(self.state["results"]) == len(self.state["tasks"]) else "paused"
            if on_checkpoint:
                on_checkpoint(self.to_dict())
            if result["status"] in {"provider_failed", "token_usage_unavailable"}:
                self.state["status"] = result["status"]
                break
        return self.to_dict()

    def to_dict(self):
        return deepcopy(self.state)


class CodeComparisonSession:
    """Run the same arriving apparatus instances with two persistent libraries."""

    def __init__(self, workspace, *, tasks=None, config=None, hegel_config=None, totals=None,
                 backend="process", state=None, provider_factory=None):
        saved = state or {}
        self.code = CodeResearchSession(workspace, tasks=tasks, config=config, totals=totals,
            backend=backend, state=saved.get("code"), provider_factory=provider_factory)
        self.config = hegel_config or ResearchConfig(**saved.get("hegel_config", {
            "observation_budget": self.code.config.observation_budget,
            "compute_budget": self.code.config.compute_budget, "max_rounds": 3}))
        if isinstance(self.config, dict):
            self.config = ResearchConfig(**self.config)
        if (self.config.observation_budget, self.config.compute_budget) != (self.code.config.observation_budget, self.code.config.compute_budget):
            raise ValueError("both arms need identical per-task observation and search caps")
        self.library = KnowledgeLibrary.from_dict(saved["hegel"]["knowledge"]) if saved else KnowledgeLibrary()
        self.hegel = deepcopy(saved.get("hegel", {"results": [], "costs": {
            "observation_cost": 0, "measurement_cost": 0, "compute_cost": 0, "python_execution_seconds": 0.}}))
        self.evaluation = deepcopy(saved.get("evaluation"))

    def add_task(self, task_index, seed):
        self.code.add_task(task_index, seed)
        self.evaluation = None

    def run(self, max_tasks=None, on_checkpoint=None):
        self.code.run(max_tasks=max_tasks, on_checkpoint=(lambda _: on_checkpoint(self.to_dict())) if on_checkpoint else None)
        target = (len(self.code.state["tasks"]) if self.code.state["status"] in {
            "completed", "budget_exhausted", "provider_failed", "token_usage_unavailable"}
            else len(self.code.state["results"]))
        return self._run_hegel_until(target, on_checkpoint)

    def run_hegel(self, max_tasks=None, on_checkpoint=None):
        """Advance the algorithm arm independently without reading final scores."""
        target = (len(self.code.state["tasks"]) if max_tasks is None else
                  min(len(self.code.state["tasks"]), len(self.hegel["results"])+max_tasks))
        return self._run_hegel_until(target, on_checkpoint)

    def _run_hegel_until(self, target, on_checkpoint):
        while len(self.hegel["results"]) < target:
            index = len(self.hegel["results"])
            item = self.code.state["tasks"][index]
            remaining = {key: self.code.state["totals"][key]-self.hegel["costs"][cost]
                         for key, cost in _BUDGET_COST.items() if key in {"observation_budget", "measurement_budget", "compute_budget"}}
            if remaining["observation_budget"] < 24 or remaining["compute_budget"] <= 0:
                self.hegel["status"] = "budget_exhausted"
                break
            config = replace(self.config, seed=item["seed"],
                observation_budget=min(self.config.observation_budget, remaining["observation_budget"]),
                compute_budget=min(self.config.compute_budget, remaining["compute_budget"]))
            started = monotonic()
            with make_lab(item["task_index"], item["seed"], backend=self.code.backend) as laboratory:
                result = MeasurementResearchAgent(config, library=self.library,
                    measurement_budget=min(self.code.config.measurement_budget, remaining["measurement_budget"])).run(laboratory)
            elapsed = monotonic()-started
            result.update(**item, task_position=index, python_execution_seconds=elapsed)
            self.hegel["results"].append(result)
            for kind in ("observation_cost", "measurement_cost", "compute_cost"):
                self.hegel["costs"][kind] += result["costs"][kind]
            self.hegel["costs"]["python_execution_seconds"] += elapsed
            if on_checkpoint:
                on_checkpoint(self.to_dict())
        return self.to_dict()

    def evaluate(self):
        if self.code.state["status"] not in {"completed", "budget_exhausted", "provider_failed", "token_usage_unavailable"} or len(self.hegel["results"]) != len(self.code.state["tasks"]):
            raise ValueError("complete both research sequences before final evaluation")
        if self.evaluation is not None:
            return deepcopy(self.evaluation)
        results = []
        for index, hegel in enumerate(self.hegel["results"]):
            code = self.code.state["results"][index] if index < len(self.code.state["results"]) else None
            item = {"task_position": index, "task_index": hegel["task_index"], "seed": hegel["seed"]}
            with make_lab(item["task_index"], item["seed"], backend=self.code.backend) as laboratory:
                for name, result in (("code", code), ("hegel", hegel)):
                    if result is None or (name == "code" and not result["model"]):
                        item[name] = {"mse": None, "error": "not_visited" if result is None else "no_completed_model",
                                      "rows": 0, "scoring_compute_cost": 0}
                        continue
                    rows = laboratory.evaluation_rows(result["selected_channels"], seed=item["seed"]+1000003, length=64, trajectories=4)
                    started = monotonic()
                    if name == "code":
                        evaluation = (evaluate_python_model(result["workspace"], result["model"], rows)
                                      if result["model"] else {"mse": None, "error": "no_completed_model"})
                        evaluation.pop("predictions", None)
                    else:
                        evaluation = {"mse": loss(Program.from_dict(result["episode"]["program"]), rows, result["episode"]["fallback"], .5)}
                    item[name] = {**evaluation, "rows": len(rows), "scoring_compute_cost": len(rows),
                                  "execution_seconds": monotonic()-started}
            results.append(item)
        self.evaluation = results
        return deepcopy(results)

    def to_dict(self):
        code = self.code.to_dict()
        code_completed = sum(bool(row["model"]) for row in code["results"])
        hegel_completed = sum(row["episode"] is not None and row["episode"]["program"] is not None
                              for row in self.hegel["results"])
        covered = len(code["results"]) == len(self.hegel["results"]) == len(code["tasks"])
        if code["status"] in {"provider_failed", "token_usage_unavailable", "budget_exhausted"}:
            status = code["status"]
        elif self.hegel.get("status") == "budget_exhausted":
            status = "budget_exhausted"
        elif covered:
            status = ("completed" if code_completed == hegel_completed == len(code["tasks"])
                      else "completed_with_failures")
        else:
            status = "paused"
        return {"experiment": "persistent_code_and_hegel_comparison", "status": status,
                "code": code, "hegel_config": asdict(self.config),
                "hegel": {**deepcopy(self.hegel), "knowledge": self.library.to_dict()},
                "evaluation": deepcopy(self.evaluation),
                "summary": {"tasks": len(code["tasks"]), "code_completed_tasks": code_completed,
                    "code_visited_tasks": len(code["results"]) + int(bool(code["active"])
                        and "task_position" not in code["active"]),
                    "code_tasks_stopped": len(code["results"]),
                    "hegel_completed_tasks": hegel_completed,
                    "hegel_visited_tasks": len(self.hegel["results"]),
                    "hegel_tasks_stopped": len(self.hegel["results"]), "code_costs": code["costs"],
                    "hegel_costs": self.hegel["costs"], "shared_files": sorted(code["knowledge"]["files"]),
                    "evaluation": deepcopy(self.evaluation)},
                "resource_contract": "Shared observation/measurement/search caps across tasks; model tokens and arbitrary Python seconds are separate resources. Evaluation follows both learning sequences."}
