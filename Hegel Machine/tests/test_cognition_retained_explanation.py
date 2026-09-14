"""An acquired explanation survives a short inconclusive research visit."""

from dataclasses import replace
import json
from random import Random

from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.selection import screen_candidates
from hegel_machine.cognition.workspace import ResearchWorkspace


def add(left, right):
    return Program("add", children=(left, right))


def square(name):
    variable = Program("feature", name=name)
    return Program("mul", children=(variable, variable))


OLD = Candidate("old_coordinate", square("x"), "composition")
JOINT = Candidate("retained_joint", add(square("x"), square("z")), "composition")
OFFSET = Candidate("local_offset", add(square("x"), Program("const", value=.1)), "refit")


class MeasuredResponse:
    """A repeatable sensor with a small unmodeled offset and an optional regime shift."""

    def __init__(self, *, changed=False):
        self.changed = changed
        self.probes = []
        self.spec = TaskSpec(
            "retained_explanation", "Predict the measured sensor response", ("x", "z"), (),
            (Probe("changed_region", tuple({"x": x, "z": 1.} for x in (-1., 0., 1.))),
             Probe("agreement_region", tuple({"x": x, "z": 0.} for x in (-1., 0., 1.)))),
            max_probe_length=3)

    def observe(self, probe):
        self.probes.append(probe)
        result, history = [], []
        for point in probe.sequence:
            target = point["x"] ** 2 + (.0 if self.changed else point["z"] ** 2) + .1
            result.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, "__target__": target})
        return result


def config(**changes):
    return replace(ResearchConfig(
        observation_budget=3, compute_budget=12000, max_rounds=1, max_candidates=8,
        strategy="fixed", use_memory=False, active_probes=False, state_search=False,
        joint_planning=False, use_warm_start=False, transition_probes=False), **changes)


def acquired(task, points, name):
    return task.observe(Probe(name, tuple({"x": x, "z": z} for x, z in points)))


def checkpoint(task, observations=None):
    if observations is None:
        observations = acquired(task, [(x, z) for x in (-2., -1., 1., 2.)
                                      for z in (-2., -1., 0., 1., 2.)], "past_training")
    reference = acquired(task, [(x, z) for x in (-1.5, -.5, .5, 1.5)
                               for z in (-2., 0., 2.)], "past_reference")
    workspace = ResearchWorkspace(task.spec.task_id, observations=observations, reference=reference,
                                  incumbent=OLD, bank=[JOINT, OFFSET], fallback=2., initial_loss=20.)
    return json.loads(json.dumps(workspace.to_dict()))


def comparisons(result):
    return [action.details for action in result.actions if action.action == "compare"]


def test_accumulated_noisy_evidence_recovers_a_larger_explanation_when_the_new_trajectory_ties():
    task = MeasuredResponse()
    task.spec = replace(task.spec, available_probes=(task.spec.available_probes[1],))
    saved = checkpoint(task)
    # The available short trajectory is one on which x squared and
    # x squared + z squared agree.
    before = ResearchWorkspace.from_dict(saved)
    result = CognitiveAgent(config()).run(task, research_state=saved)
    comparison = comparisons(result)[0]

    assert result.program == JOINT.program
    assert JOINT.program.size > OLD.program.size
    assert loss(JOINT.program, before.observations, 2., .5) > config().minimum_improvement
    assert loss(JOINT.program, before.reference, 2., .5) > config().minimum_improvement
    assert abs(comparison["fresh_gain"]) < 1e-12
    assert comparison["counterexample_gain"] is None
    assert comparison["cumulative_prediction_support"] and comparison["adopt"]
    assert comparison["decision_evidence"] == "accumulated_prediction_with_nonworse_new_trajectory"
    assert result.observation_cost == 3


def test_old_evidence_cannot_override_a_real_fresh_counterexample_to_the_joint_explanation():
    original = MeasuredResponse()
    saved = checkpoint(original)
    changed = MeasuredResponse(changed=True)
    result = CognitiveAgent(config()).run(changed, research_state=saved)
    comparison = comparisons(result)[0]
    fresh = changed.observe(changed.spec.available_probes[0])

    assert loss(JOINT.program, fresh, 2., .5) > loss(OLD.program, fresh, 2., .5) + .5
    assert comparison["best_expression"] == JOINT.program.describe()
    assert comparison["fresh_gain"] < -.5
    assert not comparison["cumulative_prediction_support"]
    assert not comparison["adopt"]
    assert result.program == OLD.program
    restored = ResearchWorkspace.from_dict(json.loads(json.dumps(result.research_state)))
    assert restored.evidence_champion.program == JOINT.program
    assert restored.incumbent.program == OLD.program


def test_json_retained_champion_is_compared_even_when_the_next_screening_sample_misses_its_support():
    original = MeasuredResponse()
    # After the first three-row experiment, the real 12-row screening sample
    # contains only the agreement region. Other acquired rows and representative
    # trajectories support the joint explanation, which still has residual .1.
    sampled = set(Random(39 + 9301).sample(range(39), 12))
    points = [(float(index % 5 - 2), 0. if index in sampled else (-2. if index % 2 else 2.))
              for index in range(36)]
    saved = checkpoint(original, acquired(original, points, "past_training"))
    changed = MeasuredResponse(changed=True)
    first = CognitiveAgent(config()).run(changed, research_state=saved)
    assert first.program == OLD.program
    state = json.loads(json.dumps(first.research_state))
    restored = ResearchWorkspace.from_dict(state)
    assert restored.evidence_champion.program == JOINT.program

    screened = screen_candidates([restored.incumbent, *restored.bank, restored.evidence_champion],
        restored.observations, restored.reference, fallback=restored.fallback, q=.5,
        budget=SearchBudget(700), complexity_weight=config().complexity_weight, maximum=2)
    assert all(row.candidate.program != JOINT.program for row in screened)

    continued = CognitiveAgent(config(max_candidates=2)).run(MeasuredResponse(), research_state=state)
    assert continued.program == JOINT.program
    assert comparisons(continued)[0]["candidates_scored"] >= 3
    assert comparisons(continued)[0]["cumulative_prediction_support"]
    assert continued.observation_cost == 3

    state["evidence_champion"] = None
    unretained = CognitiveAgent(config(max_candidates=2)).run(MeasuredResponse(), research_state=state)
    # The current screener also keeps acquired counterexamples. In this case
    # it can now recover the joint explanation from the bank even without the
    # separately retained champion that the uniform sample above missed.
    assert unretained.program == JOINT.program
    assert loss(continued.program, restored.reference, 2., .5) < loss(OLD.program, restored.reference, 2., .5)
