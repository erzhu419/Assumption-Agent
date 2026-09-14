"""The evidence and executable alternatives needed to continue one question."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .language import Program
from .models import Candidate, Observation, Probe


def observation_from_dict(payload):
    return Observation(dict(payload["features"]), payload["target"],
                       tuple(dict(row) for row in payload.get("history", ())))


def candidate_from_dict(payload):
    if payload is None:
        return None
    return Candidate(payload["candidate_id"], Program.from_dict(payload["program"]),
                     payload["operation"], payload.get("rationale", ""),
                     tuple(payload.get("parents", ())), payload.get("memory_id"))


@dataclass
class ResearchWorkspace:
    task_id: str
    observations: list[Observation] = field(default_factory=list)
    reference: list[Observation] = field(default_factory=list)
    queried: list[Probe] = field(default_factory=list)
    incumbent: Candidate | None = None
    parent: Candidate | None = None
    bank: list[Candidate] = field(default_factory=list)
    evidence_champion: Candidate | None = None
    fallback: float | None = None
    initial_loss: float | None = None
    rounds_completed: int = 0
    graph: dict = field(default_factory=dict)
    pending_actions: list[dict] = field(default_factory=list)
    method_trace: list[dict] = field(default_factory=list)
    requested_method: dict = field(default_factory=dict)
    experiment_frontier: dict = field(default_factory=dict)
    research_intent: dict = field(default_factory=dict)
    decision_frontier: dict = field(default_factory=dict)
    uncertainty_model: dict = field(default_factory=dict)
    decision_probe_conditions: list[dict] = field(default_factory=list)
    simulator_state: dict = field(default_factory=dict)

    def to_dict(self):
        # Each row retains the history returned by its own acquired trajectory.
        # Restoring observations never joins histories across experiment resets.
        def candidate(value):
            if value is None:
                return None
            return {"candidate_id": value.candidate_id, "program": value.program.to_dict(),
                    "operation": value.operation, "rationale": value.rationale,
                    "parents": list(value.parents), "memory_id": value.memory_id}

        return {"task_id": self.task_id,
                "observations": [asdict(row) for row in self.observations],
                "reference": [asdict(row) for row in self.reference],
                "queried": [asdict(probe) for probe in self.queried],
                "incumbent": candidate(self.incumbent), "parent": candidate(self.parent),
                "bank": [candidate(value) for value in self.bank],
                "evidence_champion": candidate(self.evidence_champion),
                "fallback": self.fallback, "initial_loss": self.initial_loss,
                "rounds_completed": self.rounds_completed, "graph": self.graph,
                "pending_actions": self.pending_actions, "method_trace": self.method_trace,
                "requested_method": self.requested_method,
                "experiment_frontier": self.experiment_frontier,
                "research_intent": self.research_intent,
                "decision_frontier": self.decision_frontier,
                "uncertainty_model": self.uncertainty_model,
                "decision_probe_conditions": self.decision_probe_conditions,
                "simulator_state": self.simulator_state}

    @classmethod
    def from_dict(cls, payload):
        return cls(
            task_id=payload["task_id"],
            observations=[observation_from_dict(row) for row in payload.get("observations", ())],
            reference=[observation_from_dict(row) for row in payload.get("reference", ())],
            queried=[Probe(row["probe_id"], tuple(dict(point) for point in row["sequence"]))
                     for row in payload.get("queried", ())],
            incumbent=candidate_from_dict(payload.get("incumbent")),
            parent=candidate_from_dict(payload.get("parent")),
            bank=[candidate_from_dict(row) for row in payload.get("bank", ())],
            evidence_champion=candidate_from_dict(payload.get("evidence_champion")),
            fallback=payload.get("fallback"), initial_loss=payload.get("initial_loss"),
            rounds_completed=payload.get("rounds_completed", 0),
            graph=payload.get("graph", {}), pending_actions=payload.get("pending_actions", []),
            method_trace=payload.get("method_trace", []),
            requested_method=payload.get("requested_method", {}),
            experiment_frontier=payload.get("experiment_frontier", {}),
            research_intent=payload.get("research_intent", {}),
            decision_frontier=payload.get("decision_frontier", {}),
            uncertainty_model=payload.get("uncertainty_model", {}),
            decision_probe_conditions=payload.get("decision_probe_conditions", []),
            simulator_state=payload.get("simulator_state", {}))
