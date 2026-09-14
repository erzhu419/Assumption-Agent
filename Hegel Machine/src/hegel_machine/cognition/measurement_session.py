"""Persistent research steps and tasks with selectable physical instruments."""

from copy import deepcopy
from dataclasses import asdict, replace

from .evaluation import loss
from .language import Program
from .measurement_lab import make_lab
from .measurement_research import MeasurementResearchAgent
from .memory import KnowledgeLibrary
from .models import ResearchConfig


class MeasurementSession:
    """Share learned concepts across apparatus tasks and save completed research."""

    def __init__(self, config=None, *, tasks=None, backend=None, strategy=None,
                 measurement_budget=None, evaluation_length=None,
                 evaluation_trajectories=None, state=None, library=None):
        previous = state or {}
        self.config = config or ResearchConfig(**previous.get("config", {
            "observation_budget": 128, "compute_budget": 100000,
            "max_rounds": 3, "seed": 313}))
        self.tasks = list(tasks if tasks is not None else previous.get("tasks", (0, 1, 2, 3)))
        self.backend = backend if backend is not None else previous.get("backend", "process")
        self.strategy = strategy if strategy is not None else previous.get("strategy", "learned")
        self.measurement_budget = (measurement_budget if measurement_budget is not None
                                   else previous.get("measurement_budget", 512))
        evaluation = previous.get("evaluation", {})
        self.evaluation = {
            "length": evaluation_length if evaluation_length is not None else evaluation.get("length", 64),
            "trajectories": (evaluation_trajectories if evaluation_trajectories is not None
                             else evaluation.get("trajectories", 4)),
        }
        if not self.tasks or len(set(self.tasks)) != len(self.tasks) or any(index not in range(4) for index in self.tasks):
            raise ValueError("select distinct measurement tasks from 0, 1, 2 and 3")
        if self.backend not in {"local", "process"}:
            raise ValueError("measurement backend must be local or process")
        if self.strategy not in {"learned", "controls_only", "all"}:
            raise ValueError("measurement strategy must be learned, controls_only or all")
        if self.measurement_budget < 0 or self.config.observation_budget < 24:
            raise ValueError("measurement research requires at least 24 observation rows and a nonnegative measurement budget")
        if not 1 <= self.evaluation["length"] <= 64 or self.evaluation["trajectories"] < 1:
            raise ValueError("evaluation needs positive trajectories of 1 to 64 rows")
        if state is not None:
            current_config = asdict(self.config)
            for key, value in current_config.items():
                saved = state["config"][key]
                if key in {"observation_budget", "compute_budget"}:
                    if value < saved:
                        raise ValueError(f"resume may only increase measurement session {key}")
                elif value != saved:
                    raise ValueError(f"resume with the saved measurement session config {key}")
            if self.measurement_budget < state["measurement_budget"]:
                raise ValueError("resume may only increase measurement session measurement_budget")
            for key, value in (("tasks", self.tasks), ("strategy", self.strategy),
                               ("evaluation", self.evaluation)):
                if state[key] != value:
                    raise ValueError(f"resume with the saved measurement session {key}")
            self.state = deepcopy(state)
            self.state["backend"] = self.backend
            self.state["config"] = current_config
            self.state["measurement_budget"] = self.measurement_budget
            self.library = KnowledgeLibrary.from_dict(state["knowledge"])
        else:
            self.library = library if library is not None else KnowledgeLibrary()
            self.state = {
                "experiment": "persistent_measurement_research", "checkpoint_boundary": "research_step",
                "config": asdict(self.config), "tasks": self.tasks, "backend": self.backend,
                "strategy": self.strategy, "measurement_budget": self.measurement_budget,
                "evaluation": self.evaluation, "status": "ready", "results": [],
                "knowledge": self.library.to_dict(), "completed_tasks": [], "pending_tasks": self.tasks[:],
                "total_observation_cost": 0, "total_measurement_cost": 0, "total_compute_cost": 0,
                "active_task": None,
            }

    def _update_costs(self):
        reports = [*self.state["results"]]
        if self.state.get("active_task"):
            reports.append(self.state["active_task"]["result"])
        for kind in ("observation", "measurement", "compute"):
            self.state[f"total_{kind}_cost"] = sum(report["costs"][f"{kind}_cost"] for report in reports)

    def run(self, *, max_steps=None, on_checkpoint=None, research_steps=None):
        if max_steps is not None and max_steps < 0:
            raise ValueError("steps must be nonnegative")
        if research_steps is not None and research_steps < 1:
            raise ValueError("research steps must be positive")
        if self.state["status"] == "completed":
            return deepcopy(self.state)
        steps = 0
        while self.state["pending_tasks"] and (max_steps is None or steps < max_steps):
            task_index = self.state["pending_tasks"][0]
            task_seed = self.config.seed + task_index * 10007

            def checkpoint_active(result):
                self.state["active_task"] = {"task_index": task_index, "seed": task_seed, "result": result}
                self.state["knowledge"] = self.library.to_dict()
                self.state["status"] = "paused"
                self._update_costs()
                if on_checkpoint:
                    on_checkpoint(deepcopy(self.state))

            with make_lab(task_index, task_seed, backend=self.backend) as laboratory:
                active = self.state.get("active_task")
                result = MeasurementResearchAgent(
                    replace(self.config, seed=task_seed), measurement_budget=self.measurement_budget,
                    library=self.library).run(laboratory, strategy=self.strategy,
                        research_state=active["result"]["research_state"] if active else None,
                        max_steps=research_steps, on_checkpoint=checkpoint_active)
                if result["status"] != "completed":
                    checkpoint_active(result)
                    self.state["status"] = result["status"]
                    break
                program = Program.from_dict(result["episode"]["program"])
                evaluation_seed = task_seed + 1000003
                # This stream is requested only after the research episode has
                # finished. Neither its rows nor its score enter the library.
                rows = laboratory.evaluation_rows(result["selected_channels"], seed=evaluation_seed,
                                                  **self.evaluation)
                result.update({"task_index": task_index, "seed": task_seed, "backend": self.backend,
                               "evaluation": {**self.evaluation, "seed": evaluation_seed, "rows": len(rows),
                                              "compute_cost": len(rows),
                                              "mse": loss(program, rows, result["episode"]["fallback"], .5)}})
            self.state["results"].append(result)
            self.state["active_task"] = None
            self.state["completed_tasks"].append(task_index)
            self.state["pending_tasks"].pop(0)
            self._update_costs()
            self.state["knowledge"] = self.library.to_dict()
            self.state["status"] = "paused" if self.state["pending_tasks"] else "completed"
            steps += 1
            if on_checkpoint is not None:
                on_checkpoint(deepcopy(self.state))
        if self.state["pending_tasks"] and self.state["status"] != "budget_exhausted":
            self.state["status"] = "paused"
        return deepcopy(self.state)


def session_summary(report):
    summary = {key: value for key, value in report.items() if key not in {"results", "knowledge", "active_task"}}
    active = report.get("active_task")
    summary["active_task"] = ({"task_index": active["task_index"],
        "selected_channels": active["result"]["selected_channels"],
        "stage": active["result"]["research_state"]["stage"],
        "measurement_question": active["result"]["measurement_question"],
        "costs": active["result"]["costs"]} if active else None)
    summary["results"] = [{"task_index": result["task_index"], "task_id": result["task_id"],
                           "selected_channels": result["selected_channels"],
                           "evaluation_mse": result["evaluation"]["mse"],
                           "evaluation_compute_cost": result["evaluation"]["compute_cost"],
                           "costs": result["costs"],
                           "measurement_changes": result.get("measurement_events", []),
                           "expression": result["episode"]["expression"]} for result in report["results"]]
    summary["concepts"] = len(report["knowledge"]["concepts"])
    summary["callable_definitions"] = len(report["knowledge"]["definitions"])
    return summary
