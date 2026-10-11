"""Shared objects for observations, executable hypotheses and research actions.

These objects describe learning. They do not inherit the formal certificate
track's publication or admission lifecycle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .language import Program


@dataclass(frozen=True)
class Observation:
    features: dict[str, float]
    target: float
    history: tuple[dict[str, float], ...] = ()


@dataclass(frozen=True)
class Probe:
    probe_id: str
    sequence: tuple[dict[str, float], ...]

    @property
    def cost(self) -> int:
        return len(self.sequence)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    description: str
    feature_names: tuple[str, ...]
    initial_probes: tuple[Probe, ...]
    available_probes: tuple[Probe, ...]
    risk_quantile: float = 0.5
    context: dict[str, str] = field(default_factory=dict)
    feature_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    max_probe_length: int = 12
    decision_costs: tuple[float, float] | None = None


@dataclass
class SearchBudget:
    limit: int
    spent: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.spent)

    def spend(self, units: int = 1) -> bool:
        if units < 0:
            raise ValueError("budget units must be nonnegative")
        if units > self.remaining:
            return False
        self.spent += units
        return True


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    program: Program
    operation: str
    rationale: str = ""
    parents: tuple[str, ...] = ()
    memory_id: str | None = None


@dataclass(frozen=True)
class ResearchQuestion:
    question: str
    evidence: dict[str, float]
    operations: tuple[str, ...]
    children: tuple[ResearchQuestion, ...] = ()


@dataclass
class ActionRecord:
    round: int
    action: str
    reason: str
    observation_cost: int = 0
    compute_cost: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchConfig:
    observation_budget: int = 96
    compute_budget: int = 24000
    max_rounds: int = 5
    max_candidates: int = 16
    max_question_depth: int = 2
    complexity_weight: float = 0.001
    minimum_improvement: float = 0.002
    validation_fraction: float = 0.25
    seed: int = 7
    strategy: str = "adaptive"
    use_memory: bool = True
    active_probes: bool = True
    joint_planning: bool = True
    use_abstractions: bool = True
    structural_transfer: bool = True
    synthesize_probes: bool = True
    partition: str = "development"
    proposer: str = "search"
    llm_calls_per_task: int = 2
    llm_timeout_seconds: int = 120
    llm_token_budget: int = 12000
    max_history: int = 4
    planning_depth: int = 2
    experiment_policy: str = "task_value"
    selection_policy: str = "prospective"
    use_warm_start: bool = True
    joint_search: str = "beam"
    adaptive_horizon: bool = True
    risk_learning: bool = True

    def __post_init__(self) -> None:
        if self.observation_budget < 1 or self.compute_budget < 1:
            raise ValueError("research budgets must be positive")
        if self.max_rounds < 1 or self.max_candidates < 2:
            raise ValueError("need at least one round and two candidate slots")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must lie between zero and 0.5")
        if self.strategy not in {"adaptive", "search", "fixed"}:
            raise ValueError("strategy must be adaptive, search or fixed")
        if self.partition not in {"development", "confirmation", "forward", "continual"}:
            raise ValueError("partition must be development, confirmation, forward or continual")
        if self.joint_search not in {"beam", "greedy"}:
            raise ValueError("joint_search must be beam or greedy")
        if self.experiment_policy not in {"task_value", "disagreement"}:
            raise ValueError("experiment_policy must be task_value or disagreement")
        if self.selection_policy not in {"prospective", "retrospective"}:
            raise ValueError("selection_policy must be prospective or retrospective")
        if self.proposer not in {"search", "codex", "hybrid"}:
            raise ValueError("proposer must be search, codex or hybrid")
        if self.max_history < 1 or self.planning_depth < 1:
            raise ValueError("history and planning depth must be positive")


@dataclass
class EpisodeResult:
    task_id: str
    program: Program
    fallback: float
    observations: list[Observation]
    actions: list[ActionRecord]
    questions: list[ResearchQuestion]
    observation_cost: int
    compute_cost: int
    validation_loss: float | None
    initial_loss: float
    selected_operation: str
    selected_memory_id: str | None
    concepts_added: list[str]
    memory_events: list[dict[str, Any]]
    stop_reason: str
    plans: list[dict[str, Any]] = field(default_factory=list)
    wall_time_seconds: float = 0.0
    llm_usage: dict[str, Any] = field(default_factory=dict)
    uncertainty_model: dict[str, Any] = field(default_factory=dict)
    validation_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "program": self.program.to_dict(),
            "expression": self.program.describe(),
            "program_size": self.program.size,
            "fallback": self.fallback,
            "observations_acquired": len(self.observations),
            "actions": [asdict(action) for action in self.actions],
            "questions": [asdict(question) for question in self.questions],
            "observation_cost": self.observation_cost,
            "compute_cost": self.compute_cost,
            "validation_loss": self.validation_loss,
            "initial_loss": self.initial_loss,
            "selected_operation": self.selected_operation,
            "selected_memory_id": self.selected_memory_id,
            "concepts_added": self.concepts_added,
            "memory_events": self.memory_events,
            "stop_reason": self.stop_reason,
            "plans": self.plans,
            "wall_time_seconds": self.wall_time_seconds,
            "llm_usage": self.llm_usage,
            "uncertainty_model": self.uncertainty_model,
            "validation_rows": self.validation_rows,
        }
