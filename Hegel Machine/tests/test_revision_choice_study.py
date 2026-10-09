"""Branch isolation and shared acquisition costs for the necessary-revision study."""

from collections import Counter
from dataclasses import asdict
import importlib.util
from pathlib import Path

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import _called_definitions
from hegel_machine.cognition.models import Probe, ResearchConfig
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from test_scoped_retention_comparison import LinearLab


def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_revision_choice_study.py"
    spec = importlib.util.spec_from_file_location("revision_choice_study", path)
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    return study


@pytest.mark.parametrize("profile", ("development", "confirmation"))
def test_component_changes_preserve_the_other_component_and_public_interface(profile):
    settings = {"seed": 401, "apparatus_seed": 9013, "objects": 2, "profile": profile}
    variations = {"original": {}, "change_a": {"total_weight": 1.35},
        "change_b": {"relation_radius": .45}}
    labs = {(branch, readout): UnifiedObjectLab(**settings, readout=readout, **changes)
        for branch, changes in variations.items() for readout in ("total", "event", "total_event")}
    # Distance .6 lies inside both profiles' original radius and outside .45;
    # changing the input quantities makes its event contribution observable.
    probe = Probe("component_isolation", tuple({"q_o0": first, "q_o1": second,
        "x_o0": -.3, "x_o1": .3, "u_o0": .1, "u_o1": -.05, "focus": 0.}
        for first, second in ((.7, -.4), (-.6, .9), (.2, -.8), (.8, .1))))
    batches = {key: lab.acquire(probe, ("fine",)) for key, lab in labs.items()}

    for view in ((), ("coarse",), ("fine",)):
        assert all(lab.spec_for(view) == labs["original", "total"].spec_for(view)
            for lab in labs.values())

    def verify_response_relations(rows):
        targets = {key: [row.target for row in value] for key, value in rows.items()}
        a, b, c = (targets["original", readout] for readout in ("total", "event", "total_event"))
        assert targets["change_a", "total"] == pytest.approx([1.35 * x for x in a])
        assert targets["change_a", "event"] == b
        assert [changed - old for changed, old in zip(targets["change_a", "total_event"], c)] \
            == pytest.approx([.35 * x for x in a], abs=1e-12)
        assert targets["change_b", "total"] == pytest.approx(a, abs=1e-12)
        assert targets["change_b", "event"] != pytest.approx(b, abs=1e-12)
        assert [changed - old for changed, old in zip(targets["change_b", "total_event"], c)] \
            == pytest.approx([.7 * (changed - old)
                for changed, old in zip(targets["change_b", "event"], b)], abs=1e-12)
        for readout in ("total", "event", "total_event"):
            assert [row.features for row in rows["change_a", readout]] == [
                row.features for row in rows["original", readout]]

    verify_response_relations({key: batch.observations for key, batch in batches.items()})
    for readout in ("total", "event", "total_event"):
        assert batches["change_a", readout].transitions == batches["original", readout].transitions

    # Independent evaluation must reconstruct the changed apparatus, not silently
    # fall back to default weight/radius, and must leave research charges intact.
    costs = {key: lab.costs for key, lab in labs.items()}
    terminal = {key: lab.evaluation_rows(("fine",), seed=9871, length=32, trajectories=2)
        for key, lab in labs.items()}
    verify_response_relations(terminal)
    assert {key: lab.costs for key, lab in labs.items()} == costs
    weighted = labs["change_a", "total_event"]
    saved = list(weighted.acquisitions)
    weighted.restore(saved)
    assert weighted.acquire(probe, ("fine",)).observations == batches["change_a", "total_event"].observations


def test_shared_prefix_forks_independent_research_before_any_terminal_score():
    study = runner()
    events, evaluations, paid = [], [], []

    class RecordedLinearLab(LinearLab):
        def acquire(self, probe, channels=()):
            batch = super().acquire(probe, channels)
            paid.append((self.seed, batch.observation_cost, batch.measurement_cost))
            return batch

    plan = study.protocol(seed=12901)
    plan["tasks"] = [{"coefficients": coefficients, "seed": 12901 + 10007 * position}
        for position, coefficients in enumerate(((1., 0.), (0., 1.), (1., 1.)))]
    plan["branches"] = {name: [
        {"coefficients": coefficients, "seed": 42922},
        {"coefficients": (1., 1.), "seed": 52929}]
        for name, coefficients in (("unchanged", (1., 1.)),
            ("change_a", (1.35, 1.)), ("change_b", (1., 2.)))}
    plan["research_config"] = asdict(ResearchConfig(observation_budget=96,
        compute_budget=24000, minimum_improvement=1e-7, max_rounds=1, seed=12901))
    plan["measurement_budget"] = 0
    plan["evaluation"] = {"length": 8, "trajectories": 1, "seed_offset": 1000003}
    result = study.run_study(plan, lab_factory=lambda task:
        RecordedLinearLab(**task, events=events, evaluations=evaluations))
    prefix, branches = result["prefix"], result["results"]
    assert set(branches) == {"unchanged", "change_a", "change_b"}
    sources = {source for ids in result["source_ids"].values() for source in ids}
    assert len(sources) == 3
    keys = ("observation_cost", "measurement_cost", "compute_cost")
    for branch in branches.values():
        assert set(branch["tasks"][0]["available_concepts"]) == sources
        assert branch["solved"] == 2
        assert branch["reference_evaluation_costs"] == {
            "observation_cost": 16, "measurement_cost": 0, "compute_cost": 16}
        for key in keys:
            expected = sum(row["costs"][key] for row in branch["tasks"])
            expected += branch["screening_compute_cost"] if key == "compute_cost" else 0
            assert branch["costs"][key] == expected
            assert branch["total_with_prefix"][key] == prefix["costs"][key] + expected
            assert branch["evaluation_costs"][key] == sum(
                row["evaluation_costs"][key] for row in branch["tasks"])
        for row in branch["tasks"]:
            actual_sources = {definition.source_concept_id for definition in
                _called_definitions(Program.from_dict(row["program"])) if definition.source_concept_id}
            assert set(row["nested_call_sources"]) == actual_sources

    # Three logical branch totals each include acquisition, but the shared prefix
    # is physically acquired once; compare actual lab purchases with both ledgers.
    all_rows = prefix["tasks"] + [row for branch in branches.values() for row in branch["tasks"]]
    assert len(paid) == sum(row["costs"]["queries"] for row in all_rows)
    assert sum(count for _, count, _ in paid) == sum(row["costs"]["observation_cost"] for row in all_rows)
    for row, task in zip(prefix["tasks"], plan["tasks"]):
        assert sum(seed == task["seed"] for seed, _, _ in paid) == row["costs"]["queries"]

    first_terminal = events.index("evaluation")
    assert first_terminal > 0
    assert set(events[:first_terminal]) == {"learning"}
    assert events[first_terminal:] == ["evaluation"] * 15
    expected = Counter((task["seed"], task["seed"] + 1000003, 8, 1) for task in plan["tasks"])
    for tasks in plan["branches"].values():
        for task in tasks:
            expected[(task["seed"], task["seed"] + 1000003, 8, 1)] += 2
    assert Counter(evaluations) == expected
    assert branches["unchanged"]["tasks"][0]["frozen_response_reference"]["mse"] < 1e-7
    for name in ("change_a", "change_b"):
        assert branches[name]["tasks"][0]["frozen_response_reference"]["mse"] > 1e-3
    assert result["external_model_calls"] == 0
