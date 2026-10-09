"""A measured failed response is retained or deleted without erasing its parts."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import importlib.util
from pathlib import Path
from random import Random

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.memory import KnowledgeLibrary, _called_definitions
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, TaskSpec


def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_scoped_retention_comparison.py"
    spec = importlib.util.spec_from_file_location("scoped_retention_comparison", path)
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    return study


class LinearLab:
    """All six episodes expose identical inputs; their responses are measured."""

    channels = ()

    def __init__(self, coefficients, seed, events=None, evaluations=None):
        self._coefficients, self.seed = coefficients, seed
        self._events, self._evaluations = events, evaluations
        self.spec = TaskSpec("unlabelled_linear_process", "Predict the measured response",
            ("x", "y"), (), (), feature_bounds={"x": (-1., 1.), "y": (-1., 1.)},
            max_probe_length=64)
        self.observation_cost = self.measurement_cost = 0
        self.acquisitions = []

    def spec_for(self, channels=()):
        assert not channels
        return self.spec

    def random_probe(self, seed, *, length=24, name="sample"):
        rng = Random(seed)
        return Probe(name, tuple({name: rng.uniform(*bounds)
            for name, bounds in self.spec.feature_bounds.items()} for _ in range(length)))

    def acquire(self, probe, channels=()):
        assert not channels
        if self._events is not None:
            self._events.append("learning")
        rows, history = [], []
        for point in probe.sequence:
            target = self._coefficients[0] * point["x"] + self._coefficients[1] * point["y"]
            rows.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, "__target__": target})
        self.observation_cost += len(rows)
        self.acquisitions.append({"probe_id": probe.probe_id, "sequence": list(probe.sequence),
            "channels": [], "observation_cost": len(rows), "measurement_cost": 0})
        return MeasurementBatch(rows, len(rows), 0, ())

    def evaluation_rows(self, channels=(), *, seed=1009, length=64, trajectories=4):
        if self._events is not None:
            self._events.append("evaluation")
            self._evaluations.append((self.seed, seed, length, trajectories))
        separate = LinearLab(self._coefficients, self.seed)
        return [row for index in range(trajectories)
            for row in separate.acquire(separate.random_probe(seed + 10007 * index,
                length=length, name=f"evaluation_{index}"), channels).observations]

    def restore(self, acquisitions):
        self.observation_cost = self.measurement_cost = 0
        self.acquisitions = []
        for item in acquisitions:
            self.acquire(Probe(item["probe_id"], tuple(item["sequence"])), item["channels"])

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def nested_library():
    library = KnowledgeLibrary()
    spec = LinearLab((1., 1.), 7).spec
    x, y = Program("feature", name="x"), Program("feature", name="y")
    a, _ = library.learn(Candidate("a", x, "measured"), spec)
    b, _ = library.learn(Candidate("b", y, "measured"), spec)
    da = Definition("explicit_a", ("v",), Program("arg", name="v"), a)
    db = Definition("explicit_b", ("v",), Program("arg", name="v"), b)
    ca = Program("call", name=da.definition_id, children=(x,), definition=da)
    cb = Program("call", name=db.definition_id, children=(y,), definition=db)
    joint = Program("add", children=(ca, cb))
    c, _ = library.learn(Candidate("c", joint, "compose"), spec)
    dc = Definition("explicit_c", (), joint, c)
    called = Program("call", name=dc.definition_id, definition=dc)
    child, _ = library.learn(Candidate("child", Program("mul", children=(
        Program("const", value=2.), called)), "compose"), spec)
    library.meta_priors = [{"prior_id": name, "source_concept_id": source,
        "status": "supported", "counterexamples": [], "uses": 0}
        for name, source in zip(("a", "b", "c", "child"), (a, b, c, child))]
    return library, spec, (a, b, c, child), called


def test_complete_response_identity_preserves_components_and_scoped_reassessment():
    study = runner()
    library, spec, (a, b, c, child), program = nested_library()
    before = deepcopy(library.to_dict())
    prefix = {"results": [{"research": {}}, {"research": {}}, {"research": {
        "status": "completed", "confirmation_loss": 0., "program": program.to_dict(),
        "concepts_added": [c]}}]}
    # Its executable dependencies include both valid components, but only the
    # complete old response was tested against the changed task's output.
    assert {d.source_concept_id for d in _called_definitions(program)} == {a, b, c}
    assert study._exact_response_source(prefix, library, 1e-7).concept_id == c
    challenge = {"activated": True, "source_id": c, "task_id": spec.task_id,
        "program": program.to_dict(), "loss": .5}
    conditional = study._conditional_library(library, challenge)
    marked = next(p for p in conditional.meta_priors if p["source_concept_id"] == c)
    assert marked["status"] == "conditional" and marked["counterexamples"][0]["loss"] == .5
    assert [p for p in conditional.meta_priors if p["source_concept_id"] != c] == [
        p for p in library.meta_priors if p["source_concept_id"] != c]
    assert conditional.to_dict()["concepts"] == before["concepts"]
    assert not conditional.transfer_refuted(spec, program)
    assert library.to_dict() == before

    removed = conditional.without({c})
    assert {item.concept_id for item in removed.concepts} == {a, b}
    assert {c, child} <= set(removed.deleted_ids)
    assert [item.to_dict() for item in removed.concepts] == [
        item for item in before["concepts"] if item["concept_id"] in {a, b}]
    # Shared numerical components alone cannot stand in for the exact response.
    prefix["results"][2]["research"]["concepts_added"] = [a, b]
    assert study._exact_response_source(prefix, library, 1e-7) is None


def test_three_arms_acquire_failure_before_intervening_and_finish_before_scoring():
    study = runner()
    events, evaluations = [], []
    coefficients = ((1., 0.), (0., 1.), (1., 1.), (1., 2.), (1., 1.), (1., 2.))
    plan = study.protocol(seed=12901)
    plan["tasks"] = [{"coefficients": pair, "seed": 12901 + 10007 * position}
        for position, pair in enumerate(coefficients)]
    plan["research_config"] = asdict(ResearchConfig(observation_budget=96,
        compute_budget=24000, minimum_improvement=1e-7, max_rounds=1, seed=12901))
    plan["measurement_budget"] = 0
    plan["evaluation"] = {"length": 8, "trajectories": 1, "seed_offset": 1000003}
    result = study.run_benchmark(plan, lab_factory=lambda task:
        LinearLab(**task, events=events, evaluations=evaluations))

    arms = result["results"]
    assert set(arms) == {"conditional_retention", "unmarked_retention", "global_discard"}
    source = result["source_ids"]
    assert all(len(source[name]) == 1 for name in ("a", "b", "c"))
    a, b, c = (source[name][0] for name in ("a", "b", "c"))
    assert len({a, b, c}) == 3
    challenge = result["challenge"]
    assert challenge["source_id"] == c
    assert challenge["loss"] > plan["research_config"]["minimum_improvement"]
    assert challenge["costs"]["observation_cost"] == 24
    assert challenge["costs"]["compute_cost"] > 0
    assert c not in result["initial_available_sources"]["global_discard"]
    assert {a, b} <= set(result["initial_available_sources"]["global_discard"])
    for arm in ("conditional_retention", "unmarked_retention"):
        assert {a, b, c} <= set(result["initial_available_sources"][arm])
    assert c in result["removed_concept_ids"]["global_discard"]
    assert not {a, b}.intersection(result["removed_concept_ids"]["global_discard"])

    first_evaluation = events.index("evaluation")
    assert first_evaluation > 0
    assert set(events[:first_evaluation]) == {"learning"}
    assert events[first_evaluation:] == ["evaluation"] * 18
    assert Counter(evaluations) == Counter({
        (task["seed"], task["seed"] + 1000003, 8, 1): 3 for task in plan["tasks"]})

    keys = ("observation_cost", "measurement_cost", "compute_cost")
    for arm in arms.values():
        assert len(arm["tasks"]) == 6
        assert arm["challenge_costs"] == challenge["costs"]
        assert arm["solved"] == 6
        for key in keys:
            paid = sum(task["costs"][key] for task in arm["tasks"])
            if key == "compute_cost":
                paid += arm["screening_compute_cost"]
            assert arm["costs"][key] == paid + challenge["costs"][key]
            assert arm["evaluation_costs"][key] == sum(
                task["evaluation_costs"][key] for task in arm["tasks"])
        changed = arm["tasks"][3]
        assert changed["resource_limits"] == {
            "observation_cost": 96 - challenge["costs"]["observation_cost"],
            "measurement_cost": 0, "compute_cost": 24000 - challenge["costs"]["compute_cost"]}
        assert changed["costs"]["observation_cost"] <= 96 - challenge["costs"]["observation_cost"]
        assert changed["costs"]["compute_cost"] <= 24000 - challenge["costs"]["compute_cost"]
    removed = set(result["removed_concept_ids"]["global_discard"])
    assert all(not removed.intersection(task["nested_call_sources"])
        for task in arms["global_discard"]["tasks"][3:])
    assert result["external_model_calls"] == 0
