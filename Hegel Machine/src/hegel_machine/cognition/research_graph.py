"""Persistent research branches whose evidence and intermediate programs drive work.

The diagnostic question supplies competing branches. Within a branch, each
completed step supplies executable programs to its dependent next step. A branch
that exhausts its budget keeps that frontier; a resolved question stops spending
resources until later evidence raises it again.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field

from .evaluation import initial_prediction, loss, predict
from .knowledge_evolution import bind_answers
from .language import Program
from .models import Candidate, Observation, SearchBudget
from .planning import RevisionPlan, revision_plans
from .synthesis import synthesize


@dataclass
class ResearchNode:
    """A question whose answer is an executable input to dependent questions."""

    question_id: str
    question: str
    operations: tuple[str, ...] = ("refit", "composition")
    depends_on: list[str] = field(default_factory=list)
    required_measurements: list[str] = field(default_factory=list)
    expression: Program | None = None
    observations: list[Observation] = field(default_factory=list)
    answer: Program | None = None
    alternatives: list[Program] = field(default_factory=list)
    status: str = "pending"
    loss: float | None = None
    compute_cost: int = 0
    residual_base: Program | None = None

    def to_dict(self):
        return {**self.__dict__, "operations": list(self.operations),
                "expression": self.expression.to_dict() if self.expression is not None else None,
                "answer": self.answer.to_dict() if self.answer is not None else None,
                "residual_base": self.residual_base.to_dict() if self.residual_base is not None else None,
                "alternatives": [program.to_dict() for program in self.alternatives],
                "observations": [asdict(row) for row in self.observations]}

    @classmethod
    def from_dict(cls, payload):
        value = dict(payload)
        value["operations"] = tuple(value["operations"])
        for name in ("expression", "answer"):
            value[name] = Program.from_dict(value[name]) if value[name] is not None else None
        value["residual_base"] = Program.from_dict(value["residual_base"]) if value.get("residual_base") else None
        value["alternatives"] = [Program.from_dict(program) for program in value["alternatives"]]
        value["observations"] = [Observation(dict(row["features"]), row["target"],
                                  tuple(dict(point) for point in row["history"])) for row in value["observations"]]
        return cls(**value)


class ResearchGraph:
    def __init__(self):
        self.question = None
        self.evidence_count = 0
        self.branches = {}
        self.active_order = []
        self.events = []
        self.nodes = {}

    @classmethod
    def from_dict(cls, payload):
        graph = cls()
        if not payload:
            return graph
        graph.question = deepcopy(payload.get("question"))
        graph.evidence_count = payload.get("evidence_count", 0)
        graph.active_order = list(payload.get("active_order", []))
        graph.events = deepcopy(payload.get("events", []))
        graph.nodes = {key: ResearchNode.from_dict(value) for key, value in payload.get("nodes", {}).items()}
        for key, value in payload.get("branches", {}).items():
            branch = deepcopy(value)
            branch["plan"] = RevisionPlan.from_dict(value["plan"])
            graph.branches[key] = branch
        return graph

    def to_dict(self):
        return deepcopy({
            "question": self.question, "evidence_count": self.evidence_count,
            "active_order": self.active_order, "events": self.events,
            "nodes": {key: node.to_dict() for key, node in self.nodes.items()},
            "branches": {key: {**branch, "plan": branch["plan"].to_dict()}
                         for key, branch in self.branches.items()},
        })

    def update(self, question, *, evidence_count, joint=True, depth=2):
        """Change the executable frontier when observed residual questions change.

        Fresh evidence reopens a completed investigation only while its question
        is still relevant. A paused investigation instead resumes its saved next
        step. Repeated diagnosis of the same evidence cannot rerun completed work.
        """
        self.question = asdict(question)
        self.evidence_count = evidence_count
        proposed = revision_plans(question, joint=joint, depth=depth)
        self.active_order = [plan.plan_id for plan in proposed]
        for key, branch in self.branches.items():
            plan = branch["plan"]
            if key not in self.active_order and plan.status != "resolved":
                plan.status = "resolved"
                self.events.append({"plan_id": key, "event": "resolved",
                                    "evidence_count": evidence_count,
                                    "reason": "current evidence no longer raises this question"})
        for plan in proposed:
            key = plan.plan_id
            if key not in self.branches:
                self.branches[key] = {
                    "plan": plan, "last_attempt_evidence": evidence_count,
                    "generation": 0, "history": [],
                }
                self.events.append({"plan_id": key, "event": "created",
                                    "evidence_count": evidence_count})
                continue
            branch = self.branches[key]
            previous = branch["plan"]
            changed = previous.operations != plan.operations
            reopened = previous.status == "resolved" or (
                previous.status == "completed" and evidence_count > branch["last_attempt_evidence"])
            if changed or reopened:
                branch["history"].append(previous.to_dict())
                branch["plan"] = plan
                branch["generation"] += 1
                self.events.append({"plan_id": key, "event": "replaced" if changed else "reopened",
                                    "evidence_count": evidence_count,
                                    "reason": "dependent operations changed" if changed else "new evidence raises an unresolved question"})
            else:
                # Keep executable intermediates and the pending dependency;
                # only refresh the wording of the current research question.
                previous.question = plan.question
        return self

    def pending_plans(self):
        """Finish an existing dependency before starting another fresh branch."""
        plans = [self.branches[key]["plan"] for key in self.active_order
                 if self.branches[key]["plan"].status in {"pending", "paused"}]
        return sorted(plans, key=lambda plan: plan.next_step == 0)

    def record(self, plan):
        branch = self.branches[plan.plan_id]
        branch["plan"] = plan
        branch["last_attempt_evidence"] = self.evidence_count

    def add_question(self, question_id, question, *, operations=("refit", "composition"),
                     depends_on=(), required_measurements=(), expression=None, observations=()):
        """Add a specific child question, including its own acquired evidence.

        An expression can contain arg leaves naming dependency question IDs.
        Those leaves are replaced by measured answers before execution. Without
        an expression, the answers become coordinates for ordinary synthesis.
        """
        if question_id in self.nodes:
            raise ValueError(f"research question already exists: {question_id}")
        node = ResearchNode(question_id, question, tuple(operations), [],
                            list(dict.fromkeys(required_measurements)), expression, list(observations))
        self.nodes[question_id] = node
        try:
            self.set_dependencies(question_id, depends_on)
        except ValueError:
            del self.nodes[question_id]
            raise
        self.events.append({"question_id": question_id, "event": "question_added"})
        return node

    def _invalidate(self, question_id):
        pending, seen = [question_id], set()
        while pending:
            key = pending.pop()
            if key in seen:
                continue
            seen.add(key)
            node = self.nodes[key]
            if node.answer is not None and node.answer not in node.alternatives:
                node.alternatives.append(node.answer)
            node.answer, node.loss, node.status = None, None, "pending"
            pending.extend(other.question_id for other in self.nodes.values() if key in other.depends_on)

    def add_residual_question(self, question_id, base, observations):
        """Retain a successful computation and ask what predicts its remaining error."""
        child = self.add_question(question_id, "Explain the remaining observed error", observations=observations)
        child.residual_base = base
        self.add_question(f"{question_id}:combined", "Combine the retained mechanism and the residual answer",
            depends_on=(question_id,), expression=Program("add", children=(base, Program("arg", name=question_id))))
        return child

    def set_dependencies(self, question_id, depends_on, *, expression=None):
        """Change necessary answers and reopen downstream calculations."""
        dependencies = list(dict.fromkeys(depends_on))

        def reaches(key, wanted, visited):
            if key == wanted:
                return True
            if key in visited or key not in self.nodes:
                return False
            return any(reaches(dependency, wanted, visited | {key})
                       for dependency in self.nodes[key].depends_on)

        if any(reaches(dependency, question_id, set()) for dependency in dependencies):
            raise ValueError("research answers cannot depend on themselves")
        node = self.nodes[question_id]
        changed = node.depends_on != dependencies or (expression is not None and expression != node.expression)
        node.depends_on = dependencies
        if expression is not None:
            node.expression = expression
        if changed:
            self._invalidate(question_id)
            self.events.append({"question_id": question_id, "event": "dependencies_changed",
                                "depends_on": dependencies})
        return node

    def request_measurements(self, question_id, names):
        node = self.nodes[question_id]
        new = [name for name in names if name not in node.required_measurements]
        if new:
            node.required_measurements.extend(new)
            self._invalidate(question_id)
            self.events.append({"question_id": question_id, "event": "measurement_requested", "names": new})

    def measurement_requests(self):
        """Return actual missing measurements, grouped by the dependent question."""
        requests = []
        for node in self.nodes.values():
            if node.status == "answered":
                continue
            missing = [name for name in node.required_measurements
                       if not any(name in row.features for row in node.observations)]
            if missing:
                requests.append({"question_id": node.question_id, "measurements": missing,
                                 "question": node.question})
        return requests

    def supply_measurements(self, observations, *, question_id=None):
        """Deliver acquired rows; imaginary or declared sensor names do not unblock."""
        rows = list(observations)
        changed = []
        for node in self.nodes.values():
            if question_id is not None and node.question_id != question_id:
                continue
            if question_id is None and not node.required_measurements and node.residual_base is None:
                continue
            relevant = [row for row in rows if set(node.required_measurements).issubset(row.features)]
            if not relevant:
                continue
            self._invalidate(node.question_id)
            node.observations.extend(relevant)
            changed.append(node.question_id)
            self.events.append({"question_id": node.question_id, "event": "measurements_received",
                                "rows": len(relevant)})
        return changed

    def ready_questions(self, observations=()):
        ready = []
        for node in self.nodes.values():
            if node.status == "answered":
                continue
            if any(key not in self.nodes or self.nodes[key].answer is None for key in node.depends_on):
                node.status = "blocked"
                continue
            rows = node.observations or observations
            if not any(set(node.required_measurements).issubset(row.features) for row in rows):
                node.status = "blocked"
                continue
            if node.status == "blocked":
                node.status = "pending"
            ready.append(node)
        return ready

    def execute_ready(self, observations, budget, *, incumbent=None, risk_quantile=.5,
                      max_history=4, max_nodes=4, max_candidates=12):
        """Execute available children, then use their answers in dependent nodes.

        Answers are the best measured candidates for those questions, not global
        model adoptions. Failed or incomplete work retains previous alternatives.
        The caller still compares returned programs on its own target task.
        """
        if max_nodes < 0:
            raise ValueError("maximum executed questions must be nonnegative")
        results, attempted = [], set()
        while len(attempted) < max_nodes:
            ready = [node for node in self.ready_questions(observations) if node.question_id not in attempted]
            if not ready:
                break
            node = ready[0]
            attempted.add(node.question_id)
            rows = [row for row in (node.observations or observations)
                    if set(node.required_measurements).issubset(row.features)]
            if not rows:
                node.status = "blocked"
                continue
            start = budget.spent
            fallback = initial_prediction(rows, risk_quantile)
            if node.residual_base is not None:
                if not budget.spend(len(rows)):
                    node.status = "paused"
                    break
                rows = [Observation(dict(row.features), row.target - predict(node.residual_base, row, fallback),
                    tuple({name: value for name, value in point.items() if name != "__target__"}
                          for point in row.history)) for row in rows]
                fallback = initial_prediction(rows, risk_quantile)
            answers = {key: self.nodes[key].answer for key in node.depends_on}
            generated = []
            if node.expression is not None:
                expression = bind_answers(node.expression, answers)

                def unbound(program):
                    return program.op == "arg" or any(unbound(child) for child in program.children)

                if unbound(expression):
                    node.status = "blocked"
                    continue
                missing = expression.required_features() - set.intersection(*(set(row.features) for row in rows))
                if missing:
                    self.request_measurements(node.question_id, sorted(missing))
                    node.status = "blocked"
                    continue
                generated.append(Candidate(f"answer:{node.question_id}", expression, "research_answer",
                                           node.question, tuple(node.depends_on)))
            else:
                parent = Program("const", value=fallback) if node.residual_base is not None else incumbent or Program("const", value=fallback)
                # Leave row comparisons funded even after a broad child search.
                quota = budget.remaining // max(1, min(len(ready), max_nodes - len(attempted) + 1))
                local = SearchBudget(max(0, quota // 2))
                generated.extend(synthesize(rows, operations=node.operations, budget=local,
                    incumbent=parent, risk_quantile=risk_quantile, max_history=max_history,
                    seed_programs=tuple(answers.values())))
                budget.spend(local.spent)
            candidates = [*generated[:max_candidates],
                *[Candidate(f"previous_answer:{node.question_id}:{index}", program, "research_answer")
                  for index, program in enumerate(node.alternatives[:2])]]
            ranked = []
            for candidate in candidates:
                if not budget.spend(len(rows)):
                    break
                measured = loss(candidate.program, rows, fallback, risk_quantile)
                ranked.append((measured, candidate.program.size, candidate))
            node.compute_cost += budget.spent - start
            if ranked:
                best = min(ranked, key=lambda item: item[:2])
                node.loss, node.answer, node.status = best[0], best[2].program, "answered"
                results.append(Candidate(f"answer:{node.question_id}", node.answer, "research_answer",
                                         node.question, tuple(node.depends_on)))
            else:
                node.status = "paused"
            self.events.append({"question_id": node.question_id, "event": "answer_compared" if ranked else "answer_paused",
                                "compute_cost": budget.spent - start, "loss": node.loss,
                                "depends_on": list(node.depends_on)})
        return results
