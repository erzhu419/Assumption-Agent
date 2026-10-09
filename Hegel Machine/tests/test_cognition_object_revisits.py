"""Successor object knowledge resumes paid structural research after JSON reload."""

from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Probe, ResearchConfig
from hegel_machine.cognition.unified_continuation import paid_view
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession


CONFIG = ResearchConfig(observation_budget=384, compute_budget=360000,
    minimum_improvement=1e-7, max_rounds=6, active_probes=True, seed=221)
BASE = {"environment": "objects", "objects": 2, "readout": "event",
        "apparatus_seed": 9013, "actuator_curvature": .12}
TASKS = [
    {**BASE, "seed": 221, "research_limits": {"observation_budget": 64,
        "compute_budget": 12000, "measurement_budget": 768}},
    {**BASE, "seed": 222, "rename_prefix": "successor_"},
]
RESOURCES = ("observation_cost", "measurement_cost", "compute_cost")


def persisted(value):
    return json.loads(json.dumps(value))


def recorded_factory(purchases, projections, evaluations):
    class RecordedLab(UnifiedObjectLab):
        def acquire(self, probe, channels=()):
            batch = super().acquire(probe, channels)
            purchases.append({"seed": self.seed, "probe": probe,
                "channels": tuple(channels), "rows": list(batch.observations),
                "transitions": deepcopy(batch.transitions),
                "observation_cost": batch.observation_cost,
                "measurement_cost": batch.measurement_cost})
            return batch

        def project_rows(self, rows, source_channels, target_channels):
            projected = super().project_rows(rows, source_channels, target_channels)
            projections.append({"kind": "rows", "seed": self.seed,
                "source": tuple(source_channels), "target": tuple(target_channels),
                "before": list(rows), "after": list(projected)})
            return projected

        def project_transitions(self, transitions, source_channels, target_channels):
            projected = super().project_transitions(transitions, source_channels, target_channels)
            projections.append({"kind": "transitions", "seed": self.seed,
                "source": tuple(source_channels), "target": tuple(target_channels),
                "before": deepcopy(transitions), "after": deepcopy(projected)})
            return projected

        def evaluation_rows(self, *args, **kwargs):
            evaluations.append(self.seed)
            return super().evaluation_rows(*args, **kwargs)

    return lambda item: RecordedLab(**{key: value for key, value in item.items()
        if key not in {"environment", "research_limits"}})


@pytest.fixture(scope="module")
def object_revisited_stream(tmp_path_factory):
    purchases, projections, evaluations = [], [], []
    factory = recorded_factory(purchases, projections, evaluations)
    directory = tmp_path_factory.mktemp("object_revisits")

    def save(name, value):
        path = directory / name
        path.write_text(json.dumps(value))
        return json.loads(path.read_text())

    initial = UnifiedResearchSession(CONFIG, tasks=TASKS,
        measurement_budget=4608, lab_factory=factory)
    first = save("first.json", initial.run(max_tasks=1))
    source = UnifiedResearchSession(state=first, lab_factory=factory)
    queued = save("queued.json", source.run(max_tasks=1))
    boundary = len(purchases)
    projection_boundary = len(projections)
    resumed = UnifiedResearchSession(state=queued, lab_factory=factory)
    final = save("final.json", resumed.run(max_tasks=1))
    return {"first": first, "queued": queued, "final": final,
        "purchases": purchases, "projections": projections,
        "evaluations": evaluations, "factory": factory,
        "boundary": boundary, "projection_boundary": projection_boundary}


def test_later_renamed_object_knowledge_resumes_real_relation_research(object_revisited_stream):
    stream = object_revisited_stream
    old = stream["first"]["results"][0]["research"]
    queued, final = stream["queued"], stream["final"]
    source = queued["results"][1]["research"]
    assert old["status"] == "unresolved" and old["selected_channels"] == ["fine"]
    assert source["status"] == "completed" and source["concepts_added"]
    offer, = queued["pending_revisits"]
    assert offer["basis"] == "same_paid_object_responses"
    assert set(offer["candidate"]["source_concept_ids"]) & set(source["concepts_added"])
    assert stream["evaluations"] == []
    current = final["results"][0]["research"]
    events = current["actions"][len(old["actions"]):]
    inherited = [index for index, event in enumerate(events)
                 if event["action"] == "inherit_relation_intervals"]
    confirmations = [index for index, event in enumerate(events)
                     if event["action"] == "validate_frozen_program"]
    assert inherited and confirmations and min(inherited) < max(confirmations)
    assert any(event["action"] == "condition_inherited_relation_intervals" for event in events)
    assert current["status"] == "completed" and current["confirmation_loss"] <= 1e-7
    assert final["results"][1] == queued["results"][1]
    assert final["pending_revisits"] == []
    # A new relation explanation must face a genuinely new response trajectory;
    # its agreement on old paid rows was only the reason to resume research.
    old_sequences = [batch["probe"]["sequence"] for batch in old["batches"]]
    by_probe = {item["probe"].probe_id: item
                for item in stream["purchases"][stream["boundary"]:]}
    comparisons = [event for event in events
                   if event["action"] == "compare_object_continuation_candidates"]
    assert comparisons
    with stream["factory"](TASKS[0]) as lab:
        for event in comparisons:
            assert list(by_probe[event["probe_id"]]["probe"].sequence) not in old_sequences
            batch = next(batch for batch in current["batches"]
                         if batch["probe"]["probe_id"] == event["probe_id"])
            original = [row for rows in paid_view(lab, {"batches": [batch]},
                                                 event["baseline_channels"]) for row in rows]
            revised = [row for rows in paid_view(lab, {"batches": [batch]},
                                                event["channels"]) for row in rows]
            assert [row.target for row in original] == [row.target for row in revised]
            assert event["before_loss"] == pytest.approx(loss(
                Program.from_dict(event["baseline_program"]), original,
                event["baseline_fallback"], event["risk_quantile"]))
            assert event["after_loss"] == pytest.approx(loss(
                Program.from_dict(event["program"]), revised,
                event["fallback"], event["risk_quantile"]))
    session = UnifiedResearchSession(state=final, lab_factory=stream["factory"])
    scored = session.evaluate(length=48, trajectories=2)
    assert scored["evaluation"][0]["mse"] <= CONFIG.minimum_improvement
    assert scored["costs"] == final["costs"]


def test_paid_object_views_and_transitions_survive_resume_without_repurchase(object_revisited_stream):
    stream = object_revisited_stream
    old = stream["first"]["results"][0]["research"]
    current = stream["final"]["results"][0]["research"]
    assert current["batches"][:len(old["batches"])] == old["batches"]
    assert current["actions"][:len(old["actions"])] == old["actions"]
    graph = old["research_graph"]
    assert current["research_graph"]["events"][:len(graph["events"])] == graph["events"]
    frontier = old["object_continuation_state"]
    assert frontier["candidate_frontiers"] and frontier["refutations"]
    resumed = next(event for event in current["actions"][len(old["actions"]):]
                   if event["action"] == "resume_object_research")
    with stream["factory"](TASKS[0]) as lab:
        for saved in frontier["candidate_frontiers"]:
            view = next(view for view in lab.measurement_views
                        if list(lab.spec_for(view).feature_names) == saved["features"])
            assert saved["entries"]
            for entry in saved["entries"]:
                compared = next(item for item in resumed["alternatives"]
                    if item["program"] == entry["candidate"]["program"]
                    and tuple(item["channels"]) == view)
                rows = [row for batch in paid_view(lab, old, view) for row in batch]
                assert compared["loss"] == pytest.approx(loss(
                    Program.from_dict(compared["program"]), rows,
                    entry.get("fallback", old["fallback"]), .5))
    assert current["object_continuation_state"]["refutations"][:len(frontier["refutations"])] == frontier["refutations"]
    old_ids = {batch["probe"]["probe_id"] for batch in old["batches"]}
    paid = stream["purchases"][stream["boundary"]:]
    assert paid and all(item["seed"] == TASKS[0]["seed"] for item in paid)
    assert not old_ids & {item["probe"].probe_id for item in paid}
    assert all(tuple(batch["channels"]) in UnifiedObjectLab.measurement_views
               for batch in current["batches"])
    # The screening and resumed solver consume actual projections of fine
    # evidence; "coarse" is not a channel subset or a second physical purchase.
    projected = [item for item in stream["projections"]
                 if item["seed"] == TASKS[0]["seed"]
                 and item["source"] == ("fine",) and item["target"] == ("coarse",)
                 and item["kind"] == "rows"]
    assert projected
    assert all([row.target for row in item["before"]] ==
               [row.target for row in item["after"]] for item in projected)
    for resource in ("observation_cost", "measurement_cost"):
        assert current["visit_costs"][resource] == sum(item[resource] for item in paid)
    assert any(batch["transitions"] for batch in old["batches"])


def test_object_revisit_json_uses_the_shared_pool_once(object_revisited_stream):
    stream = object_revisited_stream
    old = stream["first"]["results"][0]["research"]
    final = stream["final"]
    current = final["results"][0]["research"]
    revisit, = final["revisits"]
    assert revisit["previous"] == old
    for resource in RESOURCES:
        assert current["costs"][resource] == old["costs"][resource] + current["visit_costs"][resource]
        assert revisit["visit_costs"][resource] == current["visit_costs"][resource]
        expected = sum(item["research"]["costs"][resource] for item in final["results"])
        if resource == "compute_cost":
            expected += final["screening_compute_cost"]
        assert final["costs"][resource] == expected
    for resource, cap in zip(RESOURCES, (448, 5376, 372000)):
        assert final["costs"][resource] <= cap
    resumed = UnifiedResearchSession(state=persisted(final), lab_factory=stream["factory"])
    count = len(stream["purchases"])
    assert persisted(resumed.run()) == final
    assert len(stream["purchases"]) == count


def test_unfunded_object_continuation_keeps_paid_frontier_without_false_completion(object_revisited_stream):
    stream = object_revisited_stream
    state = stream["queued"]
    previous = state["results"][0]["research"]
    offer, = state["pending_revisits"]
    purchases, projections, evaluations = [], [], []
    factory = recorded_factory(purchases, projections, evaluations)
    config = replace(CONFIG, observation_budget=previous["costs"]["observation_cost"],
                     compute_budget=previous["costs"]["compute_cost"])
    with factory(TASKS[0]) as lab:
        report = UnifiedResearchAgent(config,
            measurement_budget=previous["costs"]["measurement_cost"],
            library=KnowledgeLibrary.from_dict(state["knowledge"])).run(lab,
                previous=previous, continuation_candidate=deepcopy(offer["candidate"]))
    assert purchases == [] and evaluations == []
    assert report["status"] == "unresolved"
    assert report["program"] == previous["program"]
    assert report["batches"] == previous["batches"]
    assert report["research_graph"] == previous["research_graph"]
    assert all(report["visit_costs"][resource] == 0 for resource in RESOURCES)


def test_retained_coarse_incumbent_reuses_its_view_without_training_on_new_confirmation():
    """A command hypothesis fits the paid region but loses on a new trajectory."""
    def add(*terms):
        result = terms[0]
        for term in terms[1:]:
            result = Program("add", children=(result, term))
        return result

    with UnifiedObjectLab(objects=2, readout="total", apparatus_seed=9013,
                          actuator_curvature=0.) as lab:
        probe = lab.random_probe(19, length=24, name="paid_fine")
        probe = Probe(probe.probe_id, tuple({**point, "x_o0": -abs(point["x_o0"])}
                                          for point in probe.sequence))
        batch = lab.acquire(probe, ("fine",))
        old = add(Program("feature", name="measured_q_o0_o1"),
                  Program("feature", name="measured_u_o0_o1"), Program("const", value=.01))
        proposed = add(*(Program("feature", name=name)
                         for name in ("q_o0", "q_o1", "u_o0", "u_o1")),
            Program("ge", children=(Program("feature", name="x_o0"), Program("const", value=0.))))
        previous = {"status": "unresolved", "costs": {**lab.costs, "compute_cost": 48},
            "actions": [], "batches": [{"probe": asdict(probe), "channels": ["fine"],
                "observations": [asdict(row) for row in batch.observations],
                "transitions": batch.transitions}], "program": old.to_dict(),
            "selected_channels": ["coarse"], "selected_kind": "current_affine", "fallback": 0.,
            "selection_loss": .0001, "confirmation_loss": .0001}
        assessments = []

        class RecordingResearch(UnifiedResearchAgent):
            def _support(self, program, training, validation, spec, budget):
                assert program == old
                assert all(program.required_features() <= set(row.features)
                           for row in [*training, *validation])
                assert training == lab.project_rows(batch.observations, ("fine",), ("coarse",))
                assert not {tuple(sorted(row.features.items())) for row in training}.intersection(
                    tuple(sorted(row.features.items())) for row in validation)
                support = super()._support(program, training, validation, spec, budget)
                assert support["validation_loss"] == pytest.approx(loss(program, validation, 0., .5))
                assessments.append(support)
                return support

        config = ResearchConfig(observation_budget=96, compute_budget=15000, max_rounds=1,
                                seed=29, strategy="fixed", minimum_improvement=1e-7)
        report = RecordingResearch(config, measurement_budget=1000).run(lab, previous=previous,
            continuation_candidate={"program": proposed.to_dict(), "channels": []})
    resumed = next(event for event in report["actions"] if event["action"] == "resume_object_research")
    comparison = next(event for event in report["actions"]
                      if event["action"] == "compare_object_continuation_candidates")
    assert resumed["selected_channels"] == []
    assert comparison["baseline_channels"] == ["coarse"] and comparison["channels"] == []
    assert comparison["before_loss"] < comparison["after_loss"]
    assert len(assessments) == 1 and report["selected_channels"] == ["coarse"]
    assert report["costs"]["observation_cost"] == 36
    assert report["costs"]["measurement_cost"] == 180
