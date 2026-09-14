"""Research outcomes affect later work after a real library save/load boundary."""

import json

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, TaskSpec
from hegel_machine.cognition.planning import RevisionPlan
from hegel_machine.cognition.selection import AssessedCandidate, PlanAllocation


def _plans():
    return [RevisionPlan("refit", "Refit", ("refit",), steps=[{"compute_cost": 100}]),
            RevisionPlan("relation_then_composition", "Combine", ("relation", "composition"),
                         steps=[{"compute_cost": 100}])]


def _measured(plan, value):
    candidate = Candidate(f"{plan}:0:result", Program("const", value=0), "refit")
    return AssessedCandidate(candidate, value, value, value, value)


def _bind(allocation, library, name="x", risk=False):
    rows = [Observation({name: float(i)}, float(i)) for i in range(3)]
    spec = TaskSpec("unrelated_public_name", "Predict", (name,), (), (),
                    decision_costs=(2., 1.) if risk else None)
    allocation.remember(rows, spec, library.research_experience)


def test_restored_research_experience_changes_renamed_task_allocation_then_revises():
    library = KnowledgeLibrary()
    learner = PlanAllocation()
    _bind(learner, library)
    plans = _plans()
    learner.update(plans, [_measured("refit", 8), _measured("relation_then_composition", 1)], 10)
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    successor = PlanAllocation()
    _bind(successor, restored, name="sensor")
    before = successor.allocate(_plans(), 500)
    assert before[1] > before[0]
    for _ in range(4):
        successor.update(plans, [_measured("refit", 0), _measured("relation_then_composition", 15)], 10)
    after = successor.allocate(_plans(), 500)
    assert after[0] > after[1] > 0
    assert sum(after) <= 500
    assert library.research_experience != restored.research_experience


def test_risk_research_does_not_inherit_numeric_tactics_and_empty_search_is_not_success():
    library = KnowledgeLibrary()
    learner = PlanAllocation()
    _bind(learner, library)
    learner.update(_plans(), [_measured("refit", 0)], 10)
    assert learner.experience["relation_then_composition"] == 0
    risk = PlanAllocation()
    _bind(risk, library, risk=True)
    assert not risk.experience
    assert risk.allocate(_plans(), 500) == [250, 250]


def test_tactic_feedback_is_invariant_to_output_loss_units():
    allocations = []
    for scale in (1., 1e6):
        allocation = PlanAllocation()
        _bind(allocation, KnowledgeLibrary())
        allocation.update(_plans(), [_measured("refit", 8 * scale),
                                     _measured("relation_then_composition", scale)], 10 * scale)
        allocations.append(allocation.allocate(_plans(), 500))
    assert allocations[0] == allocations[1]
