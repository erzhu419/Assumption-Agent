import json

from hegel_machine.cognition.abstractions import extract_templates, fitting_shape
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec


def task(name, names=("x", "z")):
    return TaskSpec(name, "predict measured values", names, (), (), context={"system_id": name})


def feature(name):
    return Program("feature", name=name)


def op(name, left, right):
    return Program(name, children=(left, right))


def learned_g(library):
    program = op("add", op("mul", feature("x"), feature("x")), op("mul", feature("x"), feature("z")))
    rows = [Observation({"x": x, "z": z}, x * x + x * z)
            for x, z in [(-1, 0), (0, 1), (1, -1), (2, 1)]]
    source_id, _ = library.learn(Candidate("measured_g", program, "composition"), task("g_source"), rows)
    return source_id


def target_rows(*, swapped=False, raw=False):
    rows = []
    for x, z in [(-1.8, -0.7), (0.4, 1.3), (1.1, -0.8), (-0.6, 0.2), (1.8, 0.5),
                 (-1.2, 1.6), (0.7, -1.3), (-0.3, -1.8), (1.5, 1.7), (-1.7, 0.8),
                 (0.1, -0.6), (0.8, 0.9), (-0.9, -1.1), (1.2, 0.3), (-0.2, 1.8),
                 (1.9, -1.4), (-1.4, 0.4), (0.6, -0.2)]:
        g = z * z + x * z if swapped else x * x + x * z
        target = 2 * x - 0.4 * z + 1 if raw else g * g + g
        rows.append(Observation({"x": x, "z": z}, target))
    return rows


def test_many_affine_sources_and_their_guards_do_not_bury_a_useful_computation():
    library = KnowledgeLibrary()
    for index in range(24):
        source = task(f"old_{index}")
        program = op("add", op("mul", Program("const", value=index + 1), feature("x")), feature("z"))
        rows = [Observation({"x": x, "z": z}, (index + 1) * x + z)
                for x, z in [(-1, 0), (0, 1), (1, -1)]]
        source_id, _ = library.learn(Candidate(str(index), program, "refit"), source, rows)
        for context in range(3):
            library.record_counterexample(source_id, task(f"scope_{index}_{context}"), 9, 0,
                                           Observation({"x": 4 + context, "z": 4}, -10))
    wanted = learned_g(library)
    before = json.dumps(library.to_dict(), sort_keys=True)
    budget = SearchBudget(1000)
    atoms = library.select_callable_atoms(task("new"), target_rows(), budget, limit=2)
    assert any(atom.name == f"{wanted}:abstract:0:definition" for atom in atoms)
    assert all(atom.op == "call" and atom.definition.source_concept_id == wanted for atom in atoms)
    assert json.dumps(library.to_dict(), sort_keys=True) == before
    report = library.last_callable_selection
    assert report["compute_cost"] == budget.spent <= 1000
    assert report["compute_cost"] == sum(report[key] for key in ("column_compute", "screen_compute", "fit_compute"))
    assert min(row["validation_loss"] for row in report["tested_atoms"]) < 1e-8
    assert len(report["screened_atoms"]) <= 8


def test_target_evidence_selects_the_correct_role_binding_over_the_same_names():
    library = KnowledgeLibrary()
    source_id = learned_g(library)
    atoms = library.select_callable_atoms(task("new"), target_rows(swapped=True), SearchBudget(1600), limit=1)
    assert len(atoms) == 1 and atoms[0].name == f"{source_id}:abstract:0:definition"
    atom = atoms[0]
    unseen = Observation({"x": 0.7, "z": -1.6}, 0)
    assert abs(atom.predict(unseen) - (1.6 * 1.6 - 0.7 * 1.6)) < 1e-10
    assert library.last_callable_selection["selected"][0]["arguments"] == ["z", "x"]


def test_new_raw_sufficient_evidence_stops_reusing_a_previously_useful_atom():
    library = KnowledgeLibrary()
    learned_g(library)
    assert library.select_callable_atoms(task("same"), target_rows(), SearchBudget(1200))
    budget = SearchBudget(1200)
    assert library.select_callable_atoms(task("same"), target_rows(raw=True), budget) == []
    assert library.last_callable_selection["status"] == "raw_coordinates_sufficient"
    assert library.last_callable_selection["screen_compute"] == 0
    assert library.last_callable_selection["compute_cost"] == budget.spent


def test_selector_respects_small_budgets_without_changing_original_proposals():
    library = KnowledgeLibrary()
    learned_g(library)
    before = KnowledgeLibrary.from_dict(library.to_dict())
    tiny = SearchBudget(30)
    assert library.select_callable_atoms(task("new"), target_rows(), tiny) == []
    assert tiny.spent == 0
    library.select_callable_atoms(task("new"), target_rows(), SearchBudget(600))
    first, second = SearchBudget(500), SearchBudget(500)
    assert library.propose(task("new"), target_rows(), first) == before.propose(task("new"), target_rows(), second)
    assert first.spent == second.spent


def test_same_shape_parameter_variants_take_turns_with_other_structures():
    library = KnowledgeLibrary()
    for index in range(18):
        program = op("mul", Program("const", value=index + 1), op("mul", feature("x"), feature("x")))
        library.learn(Candidate(str(index), program, "composition"), task(f"quadratic_{index}"))
    wanted = learned_g(library)
    atoms = library.select_callable_atoms(task("new"), target_rows(), SearchBudget(1000), limit=1)
    assert atoms and atoms[0].name == f"{wanted}:abstract:0:definition"
    assert len(library.last_callable_selection["screened_atoms"]) < 18


def test_fitted_baseline_reuses_history_and_handles_trajectory_startup_within_budget():
    library = KnowledgeLibrary()
    learned_g(library)
    rows = []
    history = []
    for index, row in enumerate(target_rows()):
        if index % 6 == 0:
            history = []
        rows.append(Observation(row.features, row.target, tuple(history)))
        history.append({**row.features, "__target__": row.target})
    baseline = Program("lag", name="__target__", lag=2)
    budget = SearchBudget(1200)
    library.select_callable_atoms(task("history"), rows, budget, baseline_program=baseline)
    report = library.last_callable_selection
    assert report["raw_coordinates"] == ["x", "z", baseline.describe()]
    assert report["tested_atoms"]
    assert report["compute_cost"] == budget.spent <= 1200
    assert report["compute_cost"] == sum(report[key] for key in ("column_compute", "screen_compute", "fit_compute"))


def test_fitted_complete_explanation_stops_redundant_callable_search():
    library = KnowledgeLibrary()
    learned_g(library)
    g = op("add", op("mul", feature("x"), feature("x")), op("mul", feature("x"), feature("z")))
    baseline = op("add", op("mul", g, g), g)
    assert library.select_callable_atoms(task("explained"), target_rows(), SearchBudget(1200),
                                         baseline_program=baseline) == []
    assert library.last_callable_selection["status"] == "raw_coordinates_sufficient"
    assert library.last_callable_selection["screen_compute"] == 0


def test_accumulator_decay_is_structural_and_accumulators_become_reusable_subexpressions():
    accumulators = [Program("accumulate", value=decay, children=(feature("x"), Program("const", value=0)))
                    for decay in (0.3, 0.8)]
    templates = [extract_templates(program, str(index))[0] for index, program in enumerate(accumulators)]
    assert fitting_shape(templates[0], {"role_0": "x"}) != fitting_shape(templates[1], {"role_0": "x"})
    whole = op("add", accumulators[0], feature("z"))
    extracted = extract_templates(whole, "accumulated")
    assert any(template.kind == "subexpression" and template.pattern.op == "accumulate" for template in extracted)
