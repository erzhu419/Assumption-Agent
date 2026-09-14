"""Policy ablations retain feedback and assign it to the producing plan."""

import pytest

from hegel_machine.cognition import controller
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, TaskSpec
from hegel_machine.cognition.planning import RevisionPlan
from hegel_machine.cognition.selection import AssessedCandidate, PlanAllocation


def test_retrospective_selection_still_measures_and_updates_plan_and_transfer_feedback(monkeypatch):
    class Environment:
        spec = TaskSpec("feedback", "Predict the reading", ("x",),
                        (Probe("initial", tuple({"x": x} for x in (-1.0, 0.0, 1.0))),), (),
                        feature_bounds={"x": (-2.0, 2.0)})

        def observe(self, probe):
            return [Observation(dict(point), point["x"]) for point in probe.sequence]

        def evaluation_rows(self):
            raise AssertionError("Learning must not use final evaluation")

    memory = KnowledgeLibrary()
    reused = Candidate("remembered", Program("const", value=0.5), "abstract_reuse", memory_id="source")
    monkeypatch.setattr(memory, "retrieve", lambda spec, observations=None: [reused])
    feedback = []
    monkeypatch.setattr(memory, "record_transfer_result", lambda candidate, spec, **values: feedback.append(values))
    allocation = PlanAllocation()
    monkeypatch.setattr(controller, "PlanAllocation", lambda: allocation)

    def execute(plan, observations, *, budget, **kwargs):
        assert budget.spend(3)
        plan.steps.append({"compute_cost": 3, "operation": "refit"})
        return [Candidate(f"{plan.plan_id}:0:refit:3:0", Program("feature", name="x"), "refit")]

    monkeypatch.setattr(controller, "execute_revision_plan", execute)
    monkeypatch.setattr(controller, "sparse_candidates", lambda *args, **kwargs: [])
    measured = []
    rank = controller.rank_candidates

    def record_rank(*args, **kwargs):
        result = rank(*args, **kwargs)
        if kwargs.get("fresh"):
            assert kwargs["fresh_weight"] == 0.0
            for row in result:
                assert row.fresh_loss is not None
                assert row.score == pytest.approx(
                    row.evidence_loss + kwargs["complexity_weight"] * row.candidate.program.size)
            measured.extend(result)
        return result

    monkeypatch.setattr(controller, "rank_candidates", record_rank)
    result = controller.CognitiveAgent(ResearchConfig(
        observation_budget=16, compute_budget=4000, max_rounds=1,
        selection_policy="retrospective", use_abstractions=False,
        use_warm_start=False, synthesize_probes=False), memory).run(Environment())
    assert measured and allocation.experience
    assert feedback and all(item["loss"] is not None and item["baseline_loss"] is not None for item in feedback)
    assert any(action.action == "reuse_check" for action in result.actions)
    assert result.observation_cost <= 16 and result.compute_cost <= 4000


def test_refit_stage_in_uncertainty_plan_does_not_receive_refit_plan_credit():
    refit = RevisionPlan("refit", "Fit parameters", ("refit",), steps=[{"compute_cost": 100}])
    uncertainty = RevisionPlan("model_then_uncertainty", "Fit a residual model", ("refit", "robustification"),
                               steps=[{"compute_cost": 200}])

    def measured(identifier, fresh_loss):
        return AssessedCandidate(Candidate(identifier, Program("const", value=fresh_loss), "refit"),
                                 score=fresh_loss, reference_loss=fresh_loss,
                                 evidence_loss=fresh_loss, fresh_loss=fresh_loss)

    assessed = [measured("model_then_uncertainty:0:refit:48:1", 1.0),
                measured("startup:refit:0:refit:48:2", 8.0)]
    allocation = PlanAllocation()
    allocation.update([refit, uncertainty], assessed, baseline_loss=10.0)
    assert allocation.experience["refit"] == pytest.approx(2.0 / 100)
    assert allocation.experience["model_then_uncertainty"] == pytest.approx(9.0 / 200)
    assert allocation.allocate([refit, uncertainty], 100)[1] > allocation.allocate([refit, uncertainty], 100)[0]
