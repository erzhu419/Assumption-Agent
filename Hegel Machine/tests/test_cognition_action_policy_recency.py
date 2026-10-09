"""A later measured research result supersedes earlier confirmation priority."""

from hegel_machine.cognition.action_policy import ActionOption, ResearchActionPolicy
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, TaskSpec


ROWS = [Observation({"x": x}, x) for x in (-2., -1., 1., 2.)]
SPEC = TaskSpec("response", "Predict the acquired response", ("x",), (), ())
OPTIONS = [ActionOption("reuse", 0, 8), ActionOption("structure", 0, 8)]


def program(coefficient):
    return Program("mul", children=(Program("const", value=coefficient),
                                    Program("feature", name="x")))


def measured_trial(policy, before, after):
    before_loss = loss(program(before), ROWS, 0., .5)
    after_loss = loss(program(after), ROWS, 0., .5)
    feedback = policy.update("reuse", before_loss=before_loss, after_loss=after_loss,
                             observation_cost=0, compute_cost=8)
    return {"context": policy.context, "action": "reuse", "trial": feedback["trials"],
            "credited_gain": feedback["last_gain"]}, before_loss, after_loss


def confirm(experience, trial):
    origin, before_loss, after_loss = trial
    ResearchActionPolicy.revise_feedback(experience, origin, status="supported",
        before_loss=before_loss, after_loss=after_loss, compute_cost=8, evidence_rows=len(ROWS))


def choose(policy):
    return policy.choose(OPTIONS, observation_remaining=16, compute_remaining=1000)


def test_old_confirmation_does_not_promote_new_negative_trial_over_untried_direction():
    experience = {}
    policy = ResearchActionPolicy(ROWS, SPEC, experience)
    old = measured_trial(policy, 0., .7)
    confirm(experience, old)
    assert choose(policy).name == "reuse"
    assert policy.last_decision["reason"] == "confirmed_direction_before_exploration"

    current = measured_trial(policy, .7, .68)
    assert current[0]["credited_gain"] < 0
    assert policy.experience["reuse"]["expected_gain"] > 0
    assert choose(policy).name == "structure"
    assert policy.last_decision["reason"] == "untried_action"

    # Delayed feedback still belongs to its original trial.
    confirm(experience, old)
    assert choose(policy).name == "structure"


def test_latest_positive_confirmation_retains_priority_over_untried_direction():
    experience = {}
    policy = ResearchActionPolicy(ROWS, SPEC, experience)
    confirm(experience, measured_trial(policy, 0., .7))
    current = measured_trial(policy, .7, .9)
    assert current[0]["credited_gain"] > 0
    assert choose(policy).name == "structure"

    confirm(experience, current)
    assert choose(policy).name == "reuse"
    assert policy.last_decision["reason"] == "confirmed_direction_before_exploration"
