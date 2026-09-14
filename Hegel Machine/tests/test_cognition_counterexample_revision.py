"""Paid counterexamples can require a larger representation despite a fresh tie."""

from hegel_machine.cognition import controller
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, TaskSpec
from hegel_machine.cognition.selection import screen_candidates
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.workspace import ResearchWorkspace
import pytest


def _models():
    x, z = Program("feature", name="x"), Program("feature", name="z")
    old = Program("ge", children=(x, Program("const", value=0)))
    coordinate = Program("add", children=(x, Program("mul", children=(Program("const", value=.4), z))))
    new = Program("ge", children=(coordinate, Program("const", value=0)))
    return old, new


def test_new_coordinate_resolves_counterexamples_when_random_reference_misses_them(monkeypatch):
    old, new = _models()
    common = tuple({"x": float(x), "z": 0.} for x in (-1, 1) for _ in range(4))
    initial = (*common, {"x": -.1, "z": 1.}, {"x": -1., "z": 0.})

    class Environment:
        spec = TaskSpec("combination", "Predict the response", ("x", "z"),
                        (Probe("initial", initial),), (), feature_bounds={"x": (-1., 1.), "z": (-1., 1.)})

        def observe(self, probe):
            return [Observation(features, new.predict(Observation(features, 0))) for features in probe.sequence]

    def generate(observations, **kwargs):
        program = old if len(observations) == len(initial) else new
        return [Candidate("candidate", program, "composition")]

    monkeypatch.setattr(controller, "sparse_candidates", generate)
    monkeypatch.setattr(controller, "representative_probe",
                        lambda spec, *, index, **kwargs: Probe(f"representative_{index}", common))
    result = controller.CognitiveAgent(ResearchConfig(
        observation_budget=40, compute_budget=6000, max_rounds=2, use_memory=False,
        state_search=False, active_probes=False, joint_planning=False, use_warm_start=False)).run(Environment())
    assert result.program == new
    comparisons = [action.details for action in result.actions if action.action == "compare"]
    assert comparisons[-1]["fresh_gain"] == 0
    assert comparisons[-1]["counterexample_gain"] > 0
    assert comparisons[-1]["adopt"]


def test_equally_fitting_axis_variants_leave_room_for_a_joint_coordinate():
    old, new = _models()
    rows = [Observation({"x": float(x), "z": 0.}, float(x > 0)) for x in (-1, 1)]
    variants = [Candidate(f"axis_{i}", Program("ge", children=(Program("feature", name="x"),
                        Program("const", value=i / 100))), "relation") for i in range(5)]
    joint = Candidate("joint", new, "composition")
    selected = screen_candidates([*variants, joint], rows, [], fallback=.5, q=.5,
                                  budget=SearchBudget(100), complexity_weight=.001, maximum=2)
    assert len(selected) == 2
    assert any(row.candidate.program == joint.program for row in selected)


@pytest.mark.parametrize("acquired_contradiction", [True, False])
def test_a_fresh_tie_simplifies_only_when_the_acquired_evidence_is_preserved(monkeypatch, acquired_contradiction):
    simple, joint = _models()
    common = tuple({"x": float(x), "z": 0.} for x in (-1, 1) for _ in range(24))
    points = (*common, {"x": -.1, "z": 1.}, {"x": .1, "z": -1.}) if acquired_contradiction else common
    candidate = Candidate("saved_joint", joint, "composition")

    class Environment:
        spec = TaskSpec("preserve_coordinate", "Predict the response", ("x", "z"), (), (),
                        feature_bounds={"x": (-1., 1.), "z": (-1., 1.)})

        def observe(self, probe):
            return [Observation(point, joint.predict(Observation(point, 0.))) for point in probe.sequence]

    environment = Environment()
    rows = environment.observe(Probe("paid", points))
    reference = environment.observe(Probe("old_reference", common[:8]))
    state = ResearchWorkspace(environment.spec.task_id, observations=rows, reference=reference,
        incumbent=candidate, bank=[candidate], fallback=.5, initial_loss=.25)
    monkeypatch.setattr(controller, "sparse_candidates", lambda *args, **kwargs: [Candidate("axis", simple, "relation")])
    monkeypatch.setattr(controller, "synthesize", lambda *args, **kwargs: [])
    monkeypatch.setattr(controller, "representative_probe",
                        lambda spec, *, index, **kwargs: Probe(f"representative_{index}", common[:8]))
    result = controller.CognitiveAgent(ResearchConfig(observation_budget=32, compute_budget=6000,
        max_rounds=1, complexity_weight=.01, use_memory=False, state_search=False,
        active_probes=False, joint_planning=False, use_warm_start=False)).run(environment, research_state=state.to_dict())
    assert result.program == (joint if acquired_contradiction else simple)
    comparison = next(action.details for action in result.actions if action.action == "compare")
    assert comparison["fresh_gain"] == 0.
    assert comparison["adopt"] is not acquired_contradiction


@pytest.mark.parametrize("complete_alternative", [False, True])
def test_new_transition_gain_does_not_erase_three_known_starts(monkeypatch, complete_alternative):
    def boundary(value):
        return Program("ge", children=(Program("feature", name="x"), Program("const", value=value)))

    old = Program("if", children=(Program("available", lag=1), boundary(.5), boundary(.5)))
    destructive = Program("if", children=(Program("available", lag=1), boundary(0.), Program("const", value=0.)))
    complete = Program("if", children=(Program("available", lag=1), boundary(0.), boundary(.5)))

    class Environment:
        spec = TaskSpec("preserve_starts", "Predict the response", ("x",), (), (),
                        feature_bounds={"x": (-2., 2.)})

        def observe(self, probe):
            rows, history = [], []
            for point in probe.sequence:
                target = float(point["x"] >= (0. if history else .5))
                rows.append(Observation(point, target, tuple(history)))
                history.append({**point, "__target__": target})
            return rows

    environment = Environment()
    rows = [row for start in (.9, 1., 1.2) for row in environment.observe(
        Probe(f"paid_{start}", tuple({"x": x} for x in (start, -1., 1., -1., 1., -1., 1.))))]
    candidate = Candidate("saved", old, "composition")
    state = ResearchWorkspace(environment.spec.task_id, observations=rows, reference=rows,
                              incumbent=candidate, fallback=.5, initial_loss=.25)
    alternatives = [Candidate("cheap", destructive, "composition")]
    if complete_alternative:
        alternatives.append(Candidate("retained_startup", complete, "composition"))
    monkeypatch.setattr(controller, "sparse_candidates", lambda *args, **kwargs: alternatives)
    monkeypatch.setattr(controller, "synthesize", lambda *args, **kwargs: [])
    fresh = tuple({"x": x} for x in (-1., .25, 1., -1., 1., -1., 1., -1.))
    monkeypatch.setattr(controller, "representative_probe",
                        lambda spec, *, index, **kwargs: Probe(f"representative_{index}", fresh))
    result = controller.CognitiveAgent(ResearchConfig(observation_budget=32, compute_budget=6000,
        max_rounds=1, complexity_weight=.01, use_memory=False, state_search=False,
        active_probes=False, joint_planning=False, use_warm_start=False)).run(environment, research_state=state.to_dict())
    assert result.program == (complete if complete_alternative else old)
    comparison = next(action.details for action in result.actions if action.action == "compare")
    assert comparison["known_response_gain"] >= 0.
    assert comparison["adopt"] is complete_alternative
