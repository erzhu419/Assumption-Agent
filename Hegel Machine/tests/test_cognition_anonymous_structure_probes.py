"""Actual anonymous membership must determine a structural intervention."""

import json
from itertools import combinations
from pathlib import Path

import pytest

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.models import Observation, Probe, SearchBudget
from hegel_machine.cognition.structural_discovery import (
    StructuralDiscovery, boundary_probe, confirm_structure, discover_structure, distinguishing_probes,
)
from hegel_machine.cognition.structure_lab import ObjectSchema


@pytest.fixture(scope="module")
def failures():
    path = Path(__file__).parent / "fixtures/cognition_v066_anonymous_boundary.json"
    return json.loads(path.read_text())["cases"]


def _paid_inputs(case):
    rows, pairs = [], []
    for batch in case["batches"]:
        history = []
        for command, item in zip(batch["probe"]["sequence"], batch["rows"]):
            row = Observation(item["features"], item["target"], tuple(history))
            rows.append(row)
            pairs.append((Observation(command, row.target), row))
            history.append({**row.features, "__target__": row.target})
    return rows, pairs


def _lab(case, **changes):
    settings = {key: value for key, value in case["task"].items() if key != "environment"}
    return AnonymousObjectLab(**{**settings, **changes})


def _active_distances(row, schema):
    values = row.features
    return [abs(values[schema.fields[a]["x"]] - values[schema.fields[b]["x"]])
        for a, b in combinations(schema.objects, 2)
        if (values[schema.fields[a]["f"]] - values[schema.fields[b]["f"]]) *
           (values[schema.fields[a]["q"]] - values[schema.fields[b]["q"]]) != 0.]


@pytest.mark.parametrize("case_index", [0, 1, 2])
def test_paid_anonymous_failures_gain_informative_boundary_evidence(failures, case_index):
    case = failures[case_index]
    schema = ObjectSchema.from_dict(case["schema"])
    rows, pairs = _paid_inputs(case)
    initial_low, initial_high = case["initial_interval"]
    radius = initial_low + .25 * (initial_high - initial_low)
    found = discover_structure(rows, schema, budget=SearchBudget(16000),
        radius_candidates=[radius], families=(case["family"],))
    assert found.fits[0]["training_loss"] < 1e-12
    found.radius_interval = tuple(case["initial_interval"])
    with _lab(case) as lab:
        old = case["failed_boundary"]
        old_batch = lab.acquire(Probe(old["probe_id"], tuple(old["sequence"])), ("fine",))
        assert not any(initial_low < value < initial_high for row in old_batch.observations
                       for value in _active_distances(row, schema))
        for serial in range(6):
            low, high = found.radius_interval
            probe = boundary_probe(found, lab.spec, index=serial, paired_observations=pairs)
            assert probe.cost == 5
            batch = lab.acquire(probe, ("fine",))
            midpoint = (low + high) / 2
            assert any(abs(value - midpoint) < 1e-12 for row in batch.observations
                       for value in _active_distances(row, schema))
            rows.extend(batch.observations)
            pairs.extend((Observation(point, row.target), row)
                         for point, row in zip(probe.sequence, batch.observations))
            found = discover_structure(rows, schema, budget=SearchBudget(16000),
                radius_candidates=[low + fraction * (high - low) for fraction in (.25, .5, .75)],
                families=(case["family"],))
            assert found.fits[0]["training_loss"] < 1e-12
            assert found.radius_interval[1] - found.radius_interval[0] <= (high - low) / 2 + 1e-12
        fresh = lab.acquire(lab.random_probe(190711, length=64), ("fine",)).observations
        assert loss(found.candidates[0].program, fresh, 0., .5) < 1e-8


@pytest.mark.parametrize("variant,refuted", [
    ("exchange", set()), ("anchored", {"translation", "permutation"}),
    ("nonlocal", {"locality"}),
])
def test_anonymous_membership_keeps_true_symmetry_tests_and_detects_real_breaking(failures, variant, refuted):
    case = failures[0]
    schema = ObjectSchema.from_dict(case["schema"])
    _, pairs = _paid_inputs(case)
    found = StructuralDiscovery([], [], [{"kind": kind, "status": "proposed", "counterexamples": []}
        for kind in ("translation", "permutation", "locality")], schema, 0, tuple(case["initial_interval"]))
    with _lab(case, variant=variant) as lab:
        plans = distinguishing_probes(found, lab.spec, paired_observations=pairs, length=4)
        assert {plan["kind"] for plan in plans} == {"translation", "permutation", "locality"}
        experiments = [{"kind": plan["kind"], "baseline": lab.acquire(plan["baseline"], ("fine",)),
            "batch": lab.acquire(plan["probe"], ("fine",))} for plan in plans]
        assert all(len(item["baseline"].observations) == len(item["batch"].observations) == 4
                   for item in experiments)
        hypotheses = confirm_structure(found, experiments, budget=SearchBudget(1000))
        assert {item["kind"] for item in hypotheses if item["status"] == "refuted"} == refuted
        assert all(item["status"] == "supported" for item in hypotheses if item["kind"] not in refuted)


def test_without_membership_witnesses_unexecutable_symmetry_tests_remain_proposed(failures):
    case = failures[0]
    schema = ObjectSchema.from_dict(case["schema"])
    found = StructuralDiscovery([], [], [{"kind": kind, "status": "proposed", "counterexamples": []}
        for kind in ("permutation", "locality")], schema, 0, tuple(case["initial_interval"]))
    with _lab(case) as lab:
        plans = distinguishing_probes(found, lab.spec, length=4)
        assert [plan["kind"] for plan in plans] == ["translation"]
        probe = boundary_probe(found, lab.spec)
        assert probe.cost == 5
        batch = lab.acquire(probe, ("fine",))
        low, high = found.radius_interval
        assert any(low < value < high for row in batch.observations for value in _active_distances(row, schema))
        hypotheses = confirm_structure(found, [], budget=SearchBudget(100))
        assert all(item["status"] == "proposed" and not item["counterexamples"] for item in hypotheses)
