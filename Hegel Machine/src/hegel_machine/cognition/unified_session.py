"""Carry callable knowledge through a stream of publicly instrumented tasks."""

from copy import deepcopy
from dataclasses import asdict, replace

from .evaluation import loss
from .language import Program
from .memory import KnowledgeLibrary
from .models import ResearchConfig


TASK_ORDER = (0, 1, 2, 3, 4, 5, 6, 7, 3, 0, 7, 4, 2, 6, 1, 5)


def task_sequence(seed, order=TASK_ORDER):
    return [{"task_index": index, "seed": seed + 10007 * position,
             "profile": "development"} for position, index in enumerate(order)]


def laboratory_for(item):
    """Select the physical interface; mechanism choice stays inside research."""
    arguments = dict(item)
    environment = arguments.pop("environment", "scalar")
    if environment == "objects":
        from .unified_object_lab import UnifiedObjectLab
        return UnifiedObjectLab(**arguments)
    if environment == "anonymous_objects":
        from .anonymous_object_lab import AnonymousObjectLab
        return AnonymousObjectLab(**arguments)
    if environment == "dynamic_instruments":
        from .dynamic_instrument_lab import DynamicInstrumentLab
        return DynamicInstrumentLab(**arguments)
    if environment != "scalar":
        raise ValueError(f"unknown laboratory interface: {environment}")
    from .unknown_task_lab import UnknownTaskLab
    return UnknownTaskLab(**arguments)


class UnifiedResearchSession:
    """Keep research memory, with task-boundary continuation and separate scoring."""

    def __init__(self, config=None, *, tasks=None, measurement_budget=768,
                 state=None, library=None, lab_factory=None):
        if state is not None:
            self.state = deepcopy(state)
            self.config = ResearchConfig(**state["config"])
            self.library = KnowledgeLibrary.from_dict(state["knowledge"])
        else:
            self.config = config or ResearchConfig(observation_budget=192,
                compute_budget=120000, minimum_improvement=1e-7, max_rounds=4,
                active_probes=False)
            self.library = library if library is not None else KnowledgeLibrary()
            self.state = {"experiment": "unified_unknown_task_research", "status": "ready",
                "config": asdict(self.config), "measurement_budget": measurement_budget,
                "tasks": list(tasks if tasks is not None else task_sequence(self.config.seed)),
                "results": [], "evaluation": [], "knowledge": self.library.to_dict()}
        if lab_factory is None:
            lab_factory = laboratory_for
        self.lab_factory = lab_factory

    def run(self, *, max_tasks=None, on_checkpoint=None):
        from .unified_research import UnifiedResearchAgent
        if max_tasks is not None and max_tasks < 1:
            raise ValueError("task steps must be positive")
        visited = 0
        while len(self.state["results"]) < len(self.state["tasks"]):
            if max_tasks is not None and visited >= max_tasks:
                break
            position = len(self.state["results"])
            item = self.state["tasks"][position]
            available = [concept.concept_id for concept in self.library.concepts]
            with self.lab_factory(item) as laboratory:
                report = UnifiedResearchAgent(replace(self.config, seed=item["seed"]),
                    measurement_budget=self.state["measurement_budget"],
                    library=self.library).run(laboratory)
            self.state["results"].append({"task_position": position,
                "available_concepts": available, "research": report})
            visited += 1
            self.state["status"] = ("completed" if len(self.state["results"]) ==
                                    len(self.state["tasks"]) else "paused")
            if on_checkpoint:
                on_checkpoint(self.to_dict())
        return self.to_dict()

    def evaluate(self, *, length=64, trajectories=4, seed_offset=1000003):
        if len(self.state["results"]) != len(self.state["tasks"]):
            raise ValueError("finish learning before independent evaluation")
        for position in range(len(self.state["evaluation"]), len(self.state["results"])):
            item = self.state["tasks"][position]
            result = self.state["results"][position]["research"]
            with self.lab_factory(item) as laboratory:
                rows = laboratory.evaluation_rows(result["selected_channels"],
                    seed=item["seed"] + seed_offset, length=length, trajectories=trajectories)
                rate = sum(channel.cost_per_row for channel in laboratory.channels
                           if channel.name in result["selected_channels"])
                risk_quantile = laboratory.spec.risk_quantile
            program = Program.from_dict(result["program"])
            measured = loss(program, rows, result["fallback"], risk_quantile)
            score = {"task_position": position, "rows": len(rows),
                "observation_cost": len(rows), "measurement_cost": len(rows) * rate,
                "compute_cost": len(rows),
                "loss": measured, "risk_quantile": risk_quantile,
                "loss_kind": "mse" if risk_quantile == .5 else "pinball"}
            if risk_quantile == .5:
                score["mse"] = measured
            self.state["evaluation"].append(score)
        return self.to_dict()

    def to_dict(self):
        report = {**deepcopy(self.state), "knowledge": self.library.to_dict()}
        report["pending_tasks"] = list(range(len(report["results"]), len(report["tasks"])))
        report["costs"] = {key: sum(row["research"]["costs"][key] for row in report["results"])
                           for key in ("observation_cost", "measurement_cost", "compute_cost")}
        return report


def unified_summary(report):
    return {key: report[key] for key in ("status", "pending_tasks", "costs", "evaluation")} | {
        "learned_concepts": len(report["knowledge"]["concepts"]),
        "results": [{"task_position": row["task_position"],
            "status": row["research"]["status"], "costs": row["research"]["costs"],
            "selected_channels": row["research"]["selected_channels"],
            "expression": Program.from_dict(row["research"]["program"]).describe()}
            for row in report["results"]]}
