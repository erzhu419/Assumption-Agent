"""Budget exhaustion keeps the latest experimentally compared predictor."""

from dataclasses import asdict

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Probe, ResearchConfig
from hegel_machine.cognition.risk_research_actions import RiskResearchActions
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab


def test_unfunded_tail_proposal_cannot_replace_the_last_measured_choice(monkeypatch):
    old_lab = UnknownTaskLab(12, seed=73091)
    batches = []
    for index in range(2):
        probe = Probe(f"paid_{index}", tuple({"u": -.9 + .075 * step} for step in range(24)))
        acquired = old_lab.acquire(probe, ())
        batches.append({"probe": asdict(probe), "channels": [],
            "observations": [asdict(row) for row in acquired.observations], "transitions": []})
    previous = {"task_id": old_lab.spec.task_id, "status": "unresolved",
        "program": Program("const", value=0.).to_dict(), "fallback": 0.,
        "selected_kind": "initial", "selected_channels": [], "actions": [], "transfer": [],
        "batches": batches, "research_graph": {}, "noise_assessment": None,
        "costs": {"observation_cost": 48, "measurement_cost": 0, "compute_cost": 0, "queries": 2}}
    bad = Program("const", value=9999.)
    proposals = []

    def propose(self, parent, channels=(), **kwargs):
        proposals.append(parent["candidate"].program)
        return {"selected": {"candidate": Candidate("pending_bad", bad, "test_tail_proposal"),
                    "kind": "test_tail_proposal", "family_id": "T03", "fallback": 0.},
                "channels": tuple(channel.name for channel in self.lab.channels), "training": []}

    monkeypatch.setattr(RiskResearchActions, "propose", propose)
    # Isolate the continuation transition after an unresolved risk assessment;
    # both candidate selection and its independent response are still real.
    monkeypatch.setattr(UnifiedResearchAgent, "_support", lambda *args:
                        {"adequate": False, "reason": "residual_loss_shift"})
    config = ResearchConfig(observation_budget=96, compute_budget=24000,
        minimum_improvement=1e-7, active_probes=False, max_rounds=2, seed=73091,
        use_memory=False)
    laboratory = UnknownTaskLab(12, seed=73091)
    result = UnifiedResearchAgent(config, measurement_budget=192).run(laboratory,
        previous=previous, continuation_candidate={"program": previous["program"],
                                                  "channels": [], "fallback": 0.})
    measured = [event for event in result["actions"] if event["action"] == "validate_continued_program"]
    assert len(proposals) == len(measured) == 1
    assert measured[0]["loss"] > config.minimum_improvement
    assert result["status"] == "unresolved"
    assert result["program"] == measured[0]["program"]
    assert result["program"] != bad.to_dict()
    assert any(item["program"] == bad.to_dict() for item in result["continuation_state"]["alternatives"])
    assert result["visit_costs"]["observation_cost"] == (
        laboratory.observation_cost - previous["costs"]["observation_cost"]) == 48
    assert result["visit_costs"]["measurement_cost"] == laboratory.measurement_cost == 192
