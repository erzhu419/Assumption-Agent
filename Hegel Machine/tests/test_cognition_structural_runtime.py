import json

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.structural_discovery import run_structural_research
from hegel_machine.cognition.structure_lab import StructureLab


CONFIG = ResearchConfig(observation_budget=192, compute_budget=160000, max_rounds=1)


def test_research_changes_real_measurements_then_continues_core_with_structure():
    lab = StructureLab(readout="total")
    library = KnowledgeLibrary()
    result = run_structural_research(lab, config=CONFIG, library=library)
    assert result["selected_scale"] == 4
    assert all(row["scale"] == 4 for row in lab.acquisitions[2:])
    assert result["measurement_reconfiguration"]["after_cost_per_row"] == 3
    assert result["measurement_reconfiguration"]["before_cost_per_row"] == 12
    assert any(action["action"] == "resume_research" for action in result["episode"]["actions"])
    assert result["episode"]["observation_cost"] > 0
    assert 0 < result["costs"]["compute_cost"] <= CONFIG.compute_budget
    assert result["costs"]["observation_cost"] <= CONFIG.observation_budget
    assert result["costs"]["measurement_cost"] <= result["measurement_budget"]
    restored = json.loads(json.dumps(result))
    program = Program.from_dict(restored["episode"]["program"])
    fresh = lab.evaluation_rows(scale=4, seed=909, length=48)
    assert loss(program, fresh, 0., .5) < 1e-8
    assert library.structural_priors[0]["support_rows"] >= 48
    assert library.structural_priors[0]["hypotheses"]


def test_checkpoint_prior_is_applied_then_revised_when_conservation_fails():
    library = KnowledgeLibrary()
    first = run_structural_research(StructureLab(readout="total"), config=CONFIG, library=library)
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    old_hypotheses = json.loads(json.dumps(library.structural_priors[0]["hypotheses"]))
    repeated = run_structural_research(StructureLab(readout="total"), config=CONFIG, library=library)
    assert repeated["prior_assessments"][0]["status"] == "retained"
    assert repeated["discovery"]["fits"][0]["family"] == "prior_reuse"
    lab = StructureLab(readout="total", variant="sink")
    revised = run_structural_research(lab, config=CONFIG, library=library)
    prior = revised["prior_assessments"][0]
    assert prior["status"] == "revised" and prior["counterexample"]
    rejected = next(item for item in prior["hypotheses_in_new_context"] if item["kind"] == "conservation")
    assert rejected["status"] == "refuted" and rejected["counterexamples"]
    assert library.structural_priors[0]["hypotheses"] == old_hypotheses
    assert first["structural_prior"]["record_id"] in revised["structural_prior"]["revises"]
    fresh = lab.evaluation_rows(scale=revised["selected_scale"], seed=191, length=48)
    current = Program.from_dict(revised["episode"]["program"])
    previous = Program.from_dict(prior["program"])
    assert loss(previous, fresh, 0., .5) > .01
    assert loss(current, fresh, 0., .5) < 1e-8


def test_fine_measurement_and_boundary_research_survive_core_and_new_trajectory():
    lab = StructureLab(readout="window")
    result = run_structural_research(lab, config=CONFIG)
    assert result["selected_scale"] == 1
    assert result["boundary_refinements"]
    before, after = result["boundary_refinements"][0], result["boundary_refinements"][-1]
    assert after["after_interval"][1] - after["after_interval"][0] < (
        before["before_interval"][1] - before["before_interval"][0]) / 100
    program = Program.from_dict(json.loads(json.dumps(result["episode"]["program"])))
    assert "window_sum" in program.describe() and "neighbor_sum" in program.describe()
    assert loss(program, lab.evaluation_rows(seed=907, length=48), 0., .5) < 1e-8
