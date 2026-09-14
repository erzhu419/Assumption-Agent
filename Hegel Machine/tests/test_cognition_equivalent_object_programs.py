"""Unidentified memberships can share an executable law without becoming true."""

from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.equivalent_object_programs import (
    canonical_object_program, propose_equivalent_object_programs,
)
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.object_induction import infer_object_model
from hegel_machine.cognition.object_interventions import propose_object_probe
from hegel_machine.cognition.unified_structure import structural_probe


def _feature(name):
    return Program("feature", name=name)


def _binary(op, left, right):
    return Program(op, children=(left, right))


def _neighbor(positions):
    return Program("neighbor_sum", children=(Program("const", value=.4), *(
        _feature(name) for index, position in enumerate(positions)
        for name in (f"f{index}", position, f"q{index}"))))


def test_normalization_uses_expression_algebra_not_matching_observations():
    left, right = _feature("x"), _feature("y")
    absolute = lambda a, b: Program("abs", children=(_binary("sub", a, b),))
    one = _binary("add", _binary("mul", left, right), absolute(left, right))
    two = Program("object_sum", children=(absolute(right, left), _binary("mul", right, left)))
    assert canonical_object_program(one) == canonical_object_program(two)
    rows = [Observation({"x": value, "y": value}, value) for value in range(10)]
    assert all(left.predict(row) == right.predict(row) for row in rows)
    assert canonical_object_program(left) != canonical_object_program(right)
    assert canonical_object_program(_binary("mul", Program("const", value=1. + 1e-12), left)) \
        != canonical_object_program(left)


def test_distance_symmetry_does_not_erase_membership_in_larger_graphs():
    ordinary, swapped = _neighbor(("x0", "x1")), _neighbor(("x1", "x0"))
    canonical = canonical_object_program(ordinary)
    assert canonical == canonical_object_program(swapped)
    row = Observation({"f0": 1., "f1": 0., "x0": .1, "x1": .3, "q0": 2., "q1": 7.}, 5.)
    assert canonical.predict(row) == ordinary.predict(row) == swapped.predict(row) == 5.
    temporal = lambda program: Program("window_sum", lag=3, children=(program,))
    assert canonical_object_program(temporal(ordinary)) == canonical_object_program(temporal(swapped))
    assert canonical_object_program(_neighbor(("x0", "x1", "x2"))) \
        != canonical_object_program(_neighbor(("x1", "x0", "x2")))


def _induce(lab):
    spec, records, budget = lab.spec_for(("fine",)), [], SearchBudget(280000)
    model = infer_object_model(spec, records, budget)
    for serial in range(130):
        probe, _ = propose_object_probe(spec, model, records, serial)
        if probe is None:
            break
        batch = lab.acquire(probe, ("fine",))
        records.extend({"controls": point, **transition, "target": observation.target,
            "probe_id": probe.probe_id, "position": position,
            "history": list(probe.sequence[:position])}
            for position, (point, transition, observation) in enumerate(
                zip(probe.sequence, batch.transitions, batch.observations)))
        model = infer_object_model(spec, records, budget)
    assert not model["complete"] and model["schema"] is None
    assert all("f" in group and "x" not in group for group in model["groups"])
    assert all(len(item["quantities"]) == 2 for item in model["position_candidates"].values())
    return model


@pytest.fixture(scope="module")
def partial_model():
    return _induce(AnonymousObjectLab(objects=2, readout="focus", apparatus_seed=9013,
                                    name_seed=2917, wiring_seed=317))


def test_shared_law_fits_once_within_the_failed_streams_actual_search_allowance():
    # These are the original development stream's first 48 fine observations.
    # Its 36,760-compute allowance could not repeat the whole grammar fit for
    # both position assignments, although their best local law was identical.
    lab = AnonymousObjectLab(objects=2, readout="focus", apparatus_seed=17011,
                             name_seed=63940, wiring_seed=3792, seed=20097)
    model = _induce(lab)
    spec = replace(lab.spec_for(("fine",)), context={"object_model": model})
    batches = [lab.acquire(lab.random_probe(20097 + 1009 * serial, length=24), ("fine",))
               for serial in (16, 17)]
    training = [row for batch in batches for row in batch.observations]
    transitions = [row for batch in batches for row in batch.transitions]
    budget = SearchBudget(36760)
    entries = propose_equivalent_object_programs(spec, training, budget, transitions=transitions)
    assert entries and entries[0]["training_loss"] < 1e-7
    assert entries[0]["structure"]["fits"][0]["family"] == "local_exchange"
    assert entries[0]["object_equivalence"]["fit_evaluations"] == 1
    assert entries[0]["object_equivalence"]["equivalence_checks"] == 2
    assert budget.spent < 24000


@pytest.mark.parametrize("readout", ["focus", "event", "window"])
def test_real_anonymous_two_object_law_refits_and_replays_with_both_assignments(partial_model, readout):
    lab = AnonymousObjectLab(objects=2, readout=readout, apparatus_seed=9013,
                             name_seed=2917, wiring_seed=317)
    spec = replace(lab.spec_for(("fine",)), context={"object_model": partial_model})
    original = json.dumps(partial_model, sort_keys=True)
    training = lab.acquire(lab.random_probe(89, length=48), ("fine",)).observations
    budget = SearchBudget(180000)
    entries = propose_equivalent_object_programs(spec, training, budget)
    assert entries and entries[0]["training_loss"] < 1e-8
    selected = entries[0]
    for serial in range(3):
        parent = json.loads(json.dumps({key: selected[key] for key in ("structure", "object_equivalence")}))
        probe = structural_probe(parent, spec, serial, 6)
        assert probe is not None
        training.extend(lab.acquire(probe, ("fine",)).observations)
        selected = propose_equivalent_object_programs(spec, training, budget, parent=parent)[0]
        assert selected["training_loss"] < 1e-8
        assert selected["structure_search"] in {"boundary_revision", "family_revision"}
        assert len(selected["object_equivalence"]["schemas"]) == 2
    fresh = lab.acquire(lab.random_probe(17017, length=64), ("fine",)).observations
    assert loss(selected["candidate"].program, fresh, 0., .5) < 1e-7
    canonical = Program.from_dict(json.loads(json.dumps(selected["object_equivalence"]["canonical_program"])))
    assert loss(canonical, fresh, 0., .5) < 1e-7
    library = KnowledgeLibrary()
    source, _ = library.learn(selected["candidate"], spec, training)
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    atom = next(call for call in restored.callable_atoms(spec, limit=64)
                if call.name == f"{source}:abstract:0:definition")
    assert atom.op == "call" and loss(atom, fresh, 0., .5) < 1e-7
    assert json.dumps(partial_model, sort_keys=True) == original
    assert not spec.context.get("object_schema")
    assert 0 < budget.spent <= budget.limit


def test_new_direct_position_response_is_not_a_shared_old_law(partial_model):
    lab = AnonymousObjectLab(objects=2, readout="focus", variant="anchored", apparatus_seed=9013,
                             name_seed=2917, wiring_seed=317)
    spec = replace(lab.spec_for(("fine",)), context={"object_model": partial_model})
    training = lab.acquire(lab.random_probe(89, length=48), ("fine",)).observations
    entries = propose_equivalent_object_programs(spec, training, SearchBudget(120000))
    assert not entries or min(entry["training_loss"] for entry in entries) > 1e-4


def test_budget_exhaustion_cannot_publish_the_first_assignment_as_equivalence(partial_model):
    lab = AnonymousObjectLab(objects=2, apparatus_seed=9013, name_seed=2917, wiring_seed=317)
    spec = replace(lab.spec_for(("fine",)), context={"object_model": partial_model})
    rows = lab.acquire(lab.random_probe(89, length=12), ("fine",)).observations
    budget = SearchBudget(2)
    assert propose_equivalent_object_programs(spec, rows, budget) == []
    assert budget.spent <= budget.limit
