from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.structure_lab import StructureLab
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_structure import propose_structures, structural_probe
from hegel_machine.cognition.unified_structure_transfer import transfer_structure


def learned_prior(readout="focus"):
    source = StructureLab(readout=readout, seed=71)
    rows = source.acquire(source.random_probe(89, length=24)).observations
    selected = propose_structures(source.spec, rows, SearchBudget(80000))[0]
    return {"source_concept_id": "concept_0014", "structure": selected["structure"]}


def test_learned_family_reopens_old_radius_on_current_short_trial():
    prior = learned_prior()
    lab = StructureLab(seed=99)
    rows = lab.acquire(lab.random_probe(317, length=8)).observations
    budget = SearchBudget(50000)
    entries = transfer_structure(lab.spec, rows, budget, prior)
    assert entries
    selected = entries[0]
    assert selected["kind"] == "structural_method_transfer:local_exchange"
    assert selected["method_source_ids"] == ["concept_0014"]
    assert selected["candidate"].parents == ("concept_0014",)
    assert selected["candidate"].memory_id == "concept_0014"
    assert selected["structure"]["radius_interval"][1] < prior["structure"]["radius_interval"][0]
    assert selected["structure"]["search_complete"]
    initial = selected["structure"]["radius_interval"]
    for serial in range(6):
        probe = structural_probe(selected, lab.spec, serial, 6)
        rows.extend(lab.acquire(probe).observations)
        selected = propose_structures(lab.spec, rows, budget, parent=selected)[0]
    current = selected["structure"]["radius_interval"]
    assert current[1] - current[0] <= (initial[1] - initial[0]) / 64 + 1e-12
    assert selected["training_loss"] < 1e-8
    fresh = lab.acquire(lab.random_probe(907, length=48)).observations
    assert loss(selected["candidate"].program, fresh, 0., .5) < 1e-8
    assert 0 < budget.spent <= budget.limit


def test_failed_learned_family_returns_no_answer_for_ordinary_research():
    prior = learned_prior("window")
    lab = StructureLab(readout="focus", seed=99)
    rows = lab.acquire(lab.random_probe(317, length=24)).observations
    budget = SearchBudget(50000)
    assert transfer_structure(lab.spec, rows, budget, prior) == []
    assert 0 < budget.spent <= budget.limit


def test_family_transfer_uses_current_instrument_roles_after_renaming():
    prior = learned_prior()
    lab = UnifiedObjectLab(seed=99, rename_prefix="next_")
    public_controls = StructureLab(seed=99)
    measured = lab.acquire(public_controls.random_probe(317, length=16), ("fine",))
    spec = lab.spec_for(("fine",))
    entries = transfer_structure(spec, measured.observations, SearchBudget(50000), prior,
                                 transitions=measured.transitions)
    assert entries
    assert entries[0]["candidate"].program.required_features().issubset(spec.feature_names)
    assert all(name.startswith("measured_next_")
               for name in entries[0]["candidate"].program.required_features())
    assert entries[0]["structure"]["schema"] != prior["structure"]["schema"]
    assert entries[0]["method_source_ids"] == ["concept_0014"]
