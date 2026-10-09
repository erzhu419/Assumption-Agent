"""An incomplete structural refit must not discard a better measured response."""

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.structural_discovery import discover_structure
from hegel_machine.cognition.structure_lab import StructureLab
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition import unified_structure


def test_short_structural_refit_retains_incumbent_on_common_acquired_evidence(monkeypatch):
    # Reproduce the production failure mode: an additional intervention is
    # bought, but its refit can fund only the first conserved-total baseline.
    original = unified_structure.propose_structures
    limited_revisions = []
    researching_view = False
    research_view = UnifiedResearchAgent._research_view

    def establish_response(self, *args, **kwargs):
        nonlocal researching_view
        researching_view = True
        try:
            return research_view(self, *args, **kwargs)
        finally:
            researching_view = False

    def incomplete_refit(spec, rows, budget, *, parent=None, **kwargs):
        if parent is None or researching_view:
            return original(spec, rows, budget, parent=parent, **kwargs)
        local = SearchBudget(min(budget.remaining, 4 * len(rows)))
        entries = original(spec, rows, local)
        budget.spend(local.spent)
        if entries:
            limited_revisions.append({"incumbent": parent["candidate"].program,
                "before": loss(parent["candidate"].program, rows, 0., spec.risk_quantile),
                "replacement": entries[0]["training_loss"]})
        return entries

    monkeypatch.setattr(unified_structure, "propose_structures", incomplete_refit)
    monkeypatch.setattr(UnifiedResearchAgent, "_research_view", establish_response)
    config = ResearchConfig(observation_budget=384, compute_budget=360000,
        minimum_improvement=1e-7, max_rounds=6, active_probes=True, seed=89)
    lab = UnifiedObjectLab(readout="window", seed=71)
    report = UnifiedResearchAgent(config, measurement_budget=4608).run(lab)
    impaired = [entry for entry in limited_revisions
                if entry["before"] + .01 < entry["replacement"]]
    assert impaired
    comparisons = [action for action in report["actions"]
                   if action["action"] == "compare_structure_revision"]
    assert comparisons
    assert all(item["selected_loss"] <= item["incumbent_loss"] + 1e-7
               for item in comparisons)
    fresh = lab.evaluation_rows(report["selected_channels"], seed=815,
                                length=32, trajectories=2)
    delivered = loss(Program.from_dict(report["program"]), fresh, report["fallback"], .5)
    measured = lab.evaluation_rows(("fine",), seed=815, length=32, trajectories=2)
    assert delivered <= min(loss(item["incumbent"], measured, 0., .5)
                            for item in impaired) + 1e-7
    assert report["costs"]["compute_cost"] <= config.compute_budget


def test_joint_response_search_keeps_the_requested_radius_candidates():
    lab = StructureLab(readout="window_total_event")
    batch = lab.acquire(lab.random_probe(89, length=48))
    budget = SearchBudget(60000)
    discovery = discover_structure(batch.observations, batch.schema, budget=budget,
        transitions=batch.transitions, radius_candidates=[0.])
    # The current-only explanations fail, so response composition must run.
    # It may revise the response basis but cannot silently reopen the range.
    assert any(fit["family"].startswith("joint_") for fit in discovery.fits)
    assert all(fit["radius"] in (None, 0.) for fit in discovery.fits)
    assert discovery.compute_cost == budget.spent <= budget.limit
