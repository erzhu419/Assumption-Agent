import json

import pytest

from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.mechanism_discovery import run_mechanism_research
from hegel_machine.cognition.mechanism_lab import MechanismLab
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Probe, ResearchConfig
from hegel_machine.cognition.response_profile import profile_prediction
from hegel_machine.cognition.structural_discovery import run_structural_research
from hegel_machine.cognition.structure_lab import StructureLab


@pytest.mark.parametrize("family", ["low_rank", "modules", "dynamics", "topology"])
def test_measured_structure_transfers_executes_and_revises_on_break(family):
    library = KnowledgeLibrary()
    source = run_mechanism_research(MechanismLab(family), library=library)
    assert source["status"] == "completed"
    assert source["experiments"][0]["status"] == "supported"
    # Restore the shared knowledge in a new process-shaped object. The next
    # episode must actually enter source definitions on new outcomes.
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    target = run_mechanism_research(MechanismLab(family), library=library,
                                    config=ResearchConfig(seed=89))
    assert target["status"] == "completed"
    assert target["discovery"]["applied_concept_ids"]
    assert set(target["discovery"]["applied_concept_ids"]) <= set(source["structural_prior"]["component_concept_ids"])
    assert target["episode"]["selected_operation"] == "mechanism_reuse"
    broken = run_mechanism_research(MechanismLab(family, "broken"), library=library)
    assert broken["status"] == "completed"
    assert broken["experiments"][0]["status"] == "refuted"
    assert any(row["counterexample"] for row in broken["prior_assessments"])
    assert source["experiments"][0]["status"] == "supported"
    # Evaluation uses an additional paid apparatus, outside the research bill.
    evaluator = MechanismLab(family, "broken")
    observed = evaluator.evaluation_rows(seed=1701)
    program = Program.from_dict(broken["episode"]["program"])
    assert sum((program.predict(row) - row.target)**2 for row in observed) / len(observed) < 1e-10
    assert evaluator.costs["observation_cost"] == len(observed)
    if family == "low_rank":
        assert source["discovery"]["rank"] == 2
        assert broken["discovery"]["rank"] == 3
        assert broken["discovery"]["reexpanded"]
    elif family == "modules":
        assert broken["discovery"]["retained_components"] == [1, 2, 3]
        assert len(source["discovery"]["modules"]) == 2
        assert len(broken["discovery"]["modules"]) == 1
    elif family == "dynamics":
        assert source["experiments"][0]["maximum_distance_ratio"] < 1
        assert broken["experiments"][0]["maximum_distance_ratio"] > 1
        assert source["discovery"]["slow_dimension"] == 2
        assert source["discovery"]["coarse_validation_loss"] < 1e-6
    else:
        assert source["discovery"]["graph"]["reachability"][0][2]
        assert not broken["discovery"]["graph"]["reachability"][0][2]


def test_order_experiment_discovers_sparse_regime_and_keeps_boundary_unknown():
    library = KnowledgeLibrary()
    source = run_mechanism_research(MechanismLab("order"), library=library)
    assert all(item["status"] == "supported" for item in source["discovery"]["hypotheses"])
    laboratory = MechanismLab("order", "broken")
    report = run_mechanism_research(laboratory, library=library)
    assert report["status"] == "completed"
    assert all(item["status"] == "refuted" for item in report["discovery"]["hypotheses"])
    assert report["experiments"][0]["points"] == [.95, .975, 1.]
    assert report["discovery"]["regimes"] == 2
    assert report["discovery"]["applied_concept_ids"] == [source["structural_prior"]["base_concept_id"]]
    interval = report["discovery"]["change_boundary"]
    assert interval["lower"] < .85 < interval["upper"]
    assert interval["upper"] - interval["lower"] < 1e-5
    features = laboratory.point([(interval["lower"] + interval["upper"]) / 2, 0, 0, 0])
    uncertain = profile_prediction(report, features)
    assert uncertain["status"] == "unknown"
    before = laboratory.costs["observation_cost"]
    payload = uncertain["probe"]
    actual = laboratory.observe(Probe(payload["probe_id"], tuple(payload["sequence"])))
    assert laboratory.costs["observation_cost"] == before + 1
    assert profile_prediction(report, features, evidence=actual) == {"status": "measured", "value": actual[0].target}
    assert profile_prediction(report, laboratory.point([.5, 0, 0, 0]))["status"] == "agreed"


def test_reexpanded_projection_survives_the_next_research_visit():
    library = KnowledgeLibrary()
    original = run_mechanism_research(MechanismLab("low_rank"), library=library)
    revised = run_mechanism_research(MechanismLab("low_rank", "broken"), library=library)
    repeated = run_mechanism_research(MechanismLab("low_rank", "broken"), library=library,
                                      config=ResearchConfig(seed=89))
    assert all(report["status"] == "completed" for report in (original, revised, repeated))
    assert original["experiments"][0]["dimension"] == 2
    assert revised["experiments"][0]["status"] == "refuted"
    assert revised["discovery"]["rank"] == 3
    assert repeated["experiments"][0]["dimension"] == 3
    assert repeated["experiments"][0]["status"] == "supported"
    assert len(repeated["experiments"][0]["tested_directions"]) == 1
    assert repeated["discovery"]["retained_components"] == [0, 1, 2, 3]
    assert repeated["discovery"]["applied_concept_ids"]
    assert repeated["episode"]["validation_loss"] < 1e-10


@pytest.mark.parametrize("family", ["low_rank", "dynamics", "order"])
def test_unfunded_evidence_stays_unresolved_and_within_all_budgets(family):
    library = KnowledgeLibrary()
    config = ResearchConfig(observation_budget=18, compute_budget=3000)
    report = run_mechanism_research(MechanismLab(family, "broken"), config, library, measurement_budget=160)
    assert report["status"] == "unresolved"
    assert report["episode"]["validation_loss"] is None
    assert report["episode"]["research_state"]["experiment_frontier"]
    assert not library.structural_priors
    assert not library.concepts
    assert report["costs"]["observation_cost"] <= 18
    assert report["costs"]["measurement_cost"] <= 160
    assert report["costs"]["compute_cost"] <= 3000


def test_mechanism_workspace_runs_in_ordinary_agent_and_mixes_with_object_priors():
    library = KnowledgeLibrary()
    report = run_mechanism_research(MechanismLab("modules"), library=library)
    agent = CognitiveAgent(ResearchConfig(observation_budget=24, compute_budget=8000, max_rounds=1), library)
    episode = agent.run(MechanismLab("modules"), research_state=report["episode"]["research_state"])
    assert episode.validation_loss < 1e-10
    assert episode.observation_cost > 0
    object_report = run_structural_research(StructureLab(readout="total"), library=library,
        config=ResearchConfig(observation_budget=120, compute_budget=64000, max_rounds=1))
    assert object_report["episode"]["validation_loss"] < 1e-8
    assert any(row.get("kind") == "mechanism" for row in library.structural_priors)
    assert any("schema" in row for row in library.structural_priors)
    for family in ("low_rank", "modules", "dynamics", "topology", "order"):
        run_mechanism_research(MechanismLab(family), library=library)
    recalled = run_structural_research(StructureLab(readout="total"), library=library,
        config=ResearchConfig(observation_budget=120, compute_budget=64000, max_rounds=1))
    assert any(item["record_id"] == object_report["structural_prior"]["record_id"]
               and item["status"] == "retained" for item in recalled["prior_assessments"])
