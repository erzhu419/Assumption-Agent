from dataclasses import replace
import json

from hegel_machine.cognition.abstractions import AbstractTemplate
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.structural_discovery import discover_structure
from hegel_machine.cognition.structure_lab import StructureLab


def test_wide_transfer_bounds_binding_generation_before_any_fit(monkeypatch):
    names = tuple(f"v{index:02}" for index in range(16))
    expression = Program("object_sum", children=tuple(Program("feature", name=name) for name in names))
    spec = TaskSpec("wide", "Predict the observed output", names, (), ())
    library = KnowledgeLibrary()
    library.learn(Candidate("source", expression, "representation"), spec)
    rows = [Observation({name: (index + 1) * .2 + column for column, name in enumerate(names)},
                        sum((index + 1) * .2 + column for column in range(16))) for index in range(8)]
    original = AbstractTemplate.mappings
    generated = 0

    def counted(self, *args, **kwargs):
        nonlocal generated
        for mapping in original(self, *args, **kwargs):
            generated += 1
            # This fails promptly for the former unbudgeted 16! enumeration.
            assert generated <= 12
            yield mapping

    monkeypatch.setattr(AbstractTemplate, "mappings", counted)
    budget = SearchBudget(96)
    candidates = library.propose(spec, rows, budget)
    assert generated == 12
    assert any(loss(candidate.program, rows, 0., .5) < 1e-8 for candidate in candidates)
    assert 0 < budget.spent <= 96


def test_object_roles_survive_checkpoint_renaming_and_real_transfer():
    lab = StructureLab()
    training = lab.acquire(lab.random_probe(89, length=24))
    learned = discover_structure(training.observations, training.schema, budget=SearchBudget(28000))
    library = KnowledgeLibrary()
    library.learn(learned.candidates[0], lab.spec, training.observations)
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    rename = {name: f"reading_{index}" for index, name in enumerate(reversed(lab.spec.feature_names))}
    schema = lab.object_schema.remap_features(rename)
    target = replace(lab.spec, task_id="new_names", feature_names=schema.feature_names,
                     context={"object_schema": json.dumps(schema.to_dict())})
    original = lab.evaluation_rows(seed=165, length=48)
    changed = [Observation({rename[name]: value for name, value in row.features.items()}, row.target,
                          tuple({rename.get(name, name): value for name, value in past.items()}
                                for past in row.history)) for row in original]
    budget = SearchBudget(1600)
    candidates = library.propose(target, changed[:24], budget)
    assert candidates
    assert min(loss(candidate.program, changed[24:], 0., .5) for candidate in candidates) < 1e-8
    whole = next(template for template in library.abstracts if template.kind == "theory")
    assert len(whole.object_roles) == len(whole.roles)
    mappings = list(whole.mappings(target.feature_names, object_schema=schema.to_dict()))
    assert len(mappings) == 24  # Four objects, not sixteen independent fields.
    atom = library.callable_atoms(target, limit=1)[0]
    assert atom.op == "call"
    assert loss(atom, changed, 0., .5) < 1e-8
    assert 0 < budget.spent <= budget.limit
