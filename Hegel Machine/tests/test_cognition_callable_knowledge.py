import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec


def spec(task, names=("x", "z")):
    return TaskSpec(task, "predict measurements", names, (), (), context={"system_id": task})


def feature(name):
    return Program("feature", name=name)


def binary(op, left, right):
    return Program(op, children=(left, right))


def source_program():
    return binary("add", binary("mul", feature("x"), feature("x")),
                  binary("mul", feature("x"), feature("z")))


def source_library():
    library = KnowledgeLibrary()
    rows = [Observation({"x": x, "z": z}, x * x + x * z)
            for x, z in [(-1, 0), (0, 1), (1, -1), (2, 1)]]
    source_id, _ = library.learn(Candidate("measured", source_program(), "composition"), spec("source"), rows)
    return library, source_id


def whole_atom(library, task, source_id):
    return next(atom for atom in library.callable_atoms(task)
                if atom.name == f"{source_id}:abstract:0:definition")


def test_learned_computation_becomes_a_role_bound_atom_and_roundtrips():
    library, source_id = source_library()
    atom = whole_atom(library, spec("new", ("u", "v")), source_id)
    assert atom.op == "call" and atom.required_features() == {"u", "v"}
    assert atom.definition.body.observed_variables() == set()
    assert len(atom.definition.arguments) == 2
    assert atom.predict(Observation({"u": 3, "v": -2}, 0)) == pytest.approx(3)
    assert atom.size == atom.expanded().size == source_program().size
    payload = json.loads(json.dumps(library.to_dict()))
    restored = KnowledgeLibrary.from_dict(payload)
    assert restored.to_dict() == payload
    assert whole_atom(restored, spec("new", ("u", "v")), source_id) == atom
    assert restored.callable_atoms(spec("new", ("u", "v")), structural_transfer=False) == []


def test_history_and_availability_are_explicit_lazy_arguments():
    library = KnowledgeLibrary()
    program = Program("if", children=(Program("available", lag=2),
                                      binary("add", Program("lag", name="x", lag=2),
                                             Program("lag", name="__target__", lag=2)),
                                      feature("x")))
    source_id, _ = library.learn(Candidate("history", program, "representation"), spec("source", ("x",)))
    atom = whole_atom(library, spec("target", ("u",)), source_id)
    assert {(child.op, child.name, child.lag) for child in atom.children} == {
        ("available", None, 2), ("lag", "u", 2), ("lag", "__target__", 2), ("feature", "u", 0)}
    assert atom.definition.body.observed_variables() == set()
    assert atom.minimum_history_depth == 0
    assert atom.predict(Observation({"u": 4}, 4)) == 4
    assert atom.predict(Observation({"u": 4}, 0, ({"u": 2, "__target__": 5}, {"u": 3, "__target__": 6}))) == 7


def test_nested_calls_keep_dependency_parents_and_deletion_removes_descendants():
    library, source_id = source_library()
    atom = whole_atom(library, spec("second"), source_id)
    second_program = binary("add", binary("mul", atom, atom), atom)
    rows = [Observation({"x": 2, "z": 1}, 42), Observation({"x": 3, "z": -2}, 12)]
    second_id, _ = library.learn(Candidate("second", second_program, "composition"), spec("second"), rows)
    second = next(concept for concept in library.concepts if concept.concept_id == second_id)
    assert source_id in second.parents and second.generation == 1
    combined = whole_atom(library, spec("third", ("u", "v")), second_id)
    assert combined.definition.body.children[0].children[0].op == "call"
    assert combined.predict(Observation({"u": 2, "v": 1}, 42)) == 42
    # Ordinary parameter templates still use paths into expanded arithmetic.
    template = next(template for template in library.abstracts if template.template_id == f"{second_id}:abstract:0")
    assert template.pattern.expanded() == template.pattern
    third_id, _ = library.learn(Candidate("third", combined, "composition"), spec("third", ("u", "v")))
    assert next(concept for concept in library.concepts if concept.concept_id == third_id).generation == 2
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    assert whole_atom(restored, spec("third", ("u", "v")), second_id).predict(Observation({"u": 2, "v": 1}, 42)) == 42
    removed = restored.without({source_id})
    assert removed.concepts == [] and removed.definitions == {}
    assert {source_id, second_id, third_id}.issubset(removed.deleted_ids)


def test_revoked_source_disables_live_calls_and_call_dependent_search_atoms():
    library, source_id = source_library()
    source_atom = whole_atom(library, spec("child"), source_id)
    child_id, _ = library.learn(Candidate("child", binary("add", source_atom, Program("const", value=2)), "composition"),
                               spec("child"))
    conflict = Observation({"x": 2, "z": 1}, -3)
    event = library.record_counterexample(source_id, spec("source"), 81, 0, conflict, budget=SearchBudget(500))
    assert event["action"] == "revoked"
    assert library.callable_atoms(spec("later")) == []
    assert all(candidate.memory_id != child_id for candidate in library.retrieve(spec("child")))
    assert library.propose(spec("later"), [conflict], SearchBudget(500)) == []


def test_same_context_restriction_exposes_guarded_call_and_no_unrestricted_theory():
    library = KnowledgeLibrary()
    task = spec("device", ("x",))
    program = binary("mul", Program("const", value=2), feature("x"))
    rows = [Observation({"x": x}, 2 * x) for x in (-1, 0, 1)]
    source_id, _ = library.learn(Candidate("original", program, "refit"), task, rows)
    library.record_counterexample(source_id, task, 36, 0, Observation({"x": 3}, 0))
    atoms = library.callable_atoms(task)
    assert atoms and all(atom.name != f"{source_id}:abstract:0:definition" for atom in atoms)
    guarded = next(atom for atom in atoms if ":condition:device:supported:" in atom.name)
    assert guarded.predict(Observation({"x": 0.5}, 1)) == 1
    assert guarded.predict(Observation({"x": 3}, 0)) == 0
    assert all(":repair:" not in atom.name for atom in atoms)


def test_changed_condition_keeps_old_and_new_call_definitions_distinct():
    library = KnowledgeLibrary()
    task = spec("device", ("x",))
    program = binary("mul", Program("const", value=2), feature("x"))
    source_id, _ = library.learn(Candidate("original", program, "refit"), task,
                                 [Observation({"x": x}, 2 * x) for x in (-1, 0, 1)])
    library.record_counterexample(source_id, task, 36, 0, Observation({"x": 3}, 0))
    old = next(atom for atom in library.callable_atoms(task) if ":supported:" in atom.name)
    library.record_counterexample(source_id, task, 25, 0, Observation({"x": 2.5}, 0))
    new = next(atom for atom in library.callable_atoms(task) if ":supported:" in atom.name)
    assert old.name != new.name and old.name not in library.definitions
    combined = binary("add", old, new)
    restored = Program.from_dict(json.loads(json.dumps(combined.to_dict())))
    assert restored.predict(Observation({"x": 0.5}, 1)) == 2


def test_many_interval_scopes_do_not_enlarge_or_reorder_the_refit_pool():
    library = KnowledgeLibrary()
    source = spec("source", ("x",))
    program = binary("mul", Program("const", value=2), feature("x"))
    source_id, _ = library.learn(Candidate("original", program, "refit"), source,
                                 [Observation({"x": x}, 2 * x) for x in (-1, 0, 1)])
    baseline = KnowledgeLibrary.from_dict(library.to_dict())
    original_templates = [template.to_dict() for template in library.abstracts]
    for index in range(5):
        task = spec(f"device_{index}", ("x",))
        library.record_counterexample(source_id, task, 36, 0, Observation({"x": 3 + index}, 0))
        supported = [Observation({"x": x}, 2 * x) for x in (-0.5, 0.5)]
        for _ in range(2):
            library.record_use(source_id, task.task_id, spec=task, observations=supported)
        atoms = library.callable_atoms(task)
        assert any(f":condition:{task.task_id}:supported:" in atom.name for atom in atoms)
    assert [template.to_dict() for template in library.abstracts] == original_templates
    unseen = spec("unseen", ("u",))
    observations = [Observation({"u": value}, 3 * value + 1) for value in (-1, 0, 1)]
    old_budget, current_budget = SearchBudget(500), SearchBudget(500)
    old = baseline.propose(unseen, observations, old_budget)
    current = library.propose(unseen, observations, current_budget)
    assert current == old
    assert current_budget.spent == old_budget.spent
    assert library.last_search == baseline.last_search
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    assert len(restored.abstracts) == len(original_templates)
    assert len(restored.definitions) == len(library.definitions)
    assert restored.callable_atoms(spec("device_2", ("x",))) == library.callable_atoms(spec("device_2", ("x",)))
