"""A paid failed confirmation must leave room to test its replacement."""

from random import Random

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab


class RepeatedResponseLab:
    channels = ()
    spec = TaskSpec("complete_confirmation", "A measured response", ("u",), (), (),
                    feature_bounds={"u": (-1., 1.)}, max_probe_length=16)

    def __init__(self, noise=0.):
        self.noise = noise
        self.acquisitions = []
        self.observations = {}

    def spec_for(self, channels=()):
        return self.spec

    def acquire(self, probe, channels=()):
        random = Random(9100 + len(self.acquisitions))
        history, rows = [], []
        for point in probe.sequence:
            target = 1. + self.noise * random.uniform(-1., 1.)
            rows.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, "__target__": target})
        self.acquisitions.append(probe)
        self.observations[probe.probe_id] = rows
        return MeasurementBatch(rows, len(rows), 0, ())


def consuming_research(agent, *, exhaust_initial=False):
    calls = []

    def research(spec, training, validation, budget, actions, transfer, graph, **kwargs):
        revised = bool(calls)
        program = Program("const", value=1. if revised else 0.)
        assert budget.spend(len(training) + len(validation))
        entry = {"candidate": Candidate(f"paid_fit_{len(calls)}", program, "fit"),
                 "kind": "current_affine", "family_id": "T02",
                 "training_loss": loss(program, training, 0., .5),
                 "validation_loss": loss(program, validation, 0., .5)}
        calls.append({"training": list(training), "validation": list(validation),
                      "limit": budget.limit, "rows_remaining": kwargs["observation_remaining"]})
        if revised or exhaust_initial:
            # Model the observed broad fallback that spends everything it is
            # offered. The outer action must fund confirmation separately.
            assert budget.spend(budget.remaining)
        return [entry]

    agent._research_view = research
    return calls


@pytest.mark.parametrize("noise,compute,risk_learning", [(0., 500, False), (.01, 26000, True)])
def test_failed_confirmation_can_score_replacement_after_research_exhausts_allowance(
        noise, compute, risk_learning):
    config = ResearchConfig(observation_budget=96, compute_budget=compute,
        max_rounds=2, max_question_depth=0, use_memory=False, active_probes=False,
        risk_learning=risk_learning, minimum_improvement=1e-7)
    agent = UnifiedResearchAgent(config)
    laboratory = RepeatedResponseLab(noise)
    calls = consuming_research(agent)

    result = agent.run(laboratory)

    assert len(calls) == 2
    validations = [event for event in result["actions"] if event["action"] == "validate_frozen_program"]
    assert len(validations) == 2
    assert not validations[0]["accepted"]
    assert validations[1]["loss"] < 1e-4
    if risk_learning:
        assert validations[1]["noise_assessment"]["reason"] != "insufficient_budget"
        assert validations[1]["noise_assessment"].get("replication")
    else:
        assert result["status"] == "completed"
    confirmation_ids = [event["probe_id"] for event in result["actions"]
                        if event["action"] == "acquire" and event["purpose"] == "confirm_frozen_program"]
    assert len(confirmation_ids) == len(set(confirmation_ids)) == 2
    # No target from the last, independent trajectory entered the revision fit.
    final_rows = laboratory.observations[confirmation_ids[-1]]
    fitted_rows = [*calls[1]["training"], *calls[1]["validation"]]
    assert not {id(row) for row in final_rows} & {id(row) for row in fitted_rows}
    assert len(calls[1]["training"]) > len(calls[0]["training"])
    assert result["costs"]["compute_cost"] <= compute
    assert result["costs"]["observation_cost"] <= config.observation_budget


def test_unfunded_scoring_does_not_purchase_unusable_confirmation():
    config = ResearchConfig(observation_budget=96, compute_budget=500,
        use_memory=False, active_probes=False, risk_learning=False, minimum_improvement=1e-7)
    agent = UnifiedResearchAgent(config)
    consuming_research(agent, exhaust_initial=True)
    result = agent.run(RepeatedResponseLab())

    assert result["status"] == "unresolved"
    assert any(event["action"] == "defer_frozen_confirmation" for event in result["actions"])
    assert not any(event["action"] == "acquire" and event["purpose"] == "confirm_frozen_program"
                   for event in result["actions"])
    assert not any(event["action"] == "validate_frozen_program" for event in result["actions"])


def test_real_numeric_research_keeps_learning_under_normal_budget():
    config = ResearchConfig(observation_budget=192, compute_budget=120000,
        minimum_improvement=1e-7, max_candidates=16, use_memory=False)
    laboratory = UnknownTaskLab(1, 113)
    result = UnifiedResearchAgent(config).run(laboratory)
    rows = laboratory.evaluation_rows(result["selected_channels"], seed=733,
                                     length=40, trajectories=2)
    assert result["status"] == "completed"
    assert loss(Program.from_dict(result["program"]), rows, result["fallback"], .5) < 1e-12


def test_revision_stops_when_complete_noise_confirmation_is_unaffordable():
    config = ResearchConfig(observation_budget=96, compute_budget=500,
        use_memory=False, active_probes=False, minimum_improvement=1e-7,
        max_question_depth=0)
    agent = UnifiedResearchAgent(config)
    calls = consuming_research(agent)
    result = agent.run(RepeatedResponseLab(.01))

    assert len(calls) == 1
    assert result["status"] == "unresolved"
    assert any(event["action"] == "defer_confirmation_revision" for event in result["actions"])
    purchased = [event for event in result["actions"] if event["action"] == "acquire"
                 and event["purpose"] == "confirm_frozen_program"]
    scored = [event for event in result["actions"] if event["action"] == "validate_frozen_program"]
    assert len(purchased) == len(scored) == 1


def test_paid_response_refit_improves_prediction_without_claiming_confirmation():
    spec = RepeatedResponseLab.spec
    training = [Observation({"u": x}, 3. * x + 2.) for x in (-1., -.5, 0., .5, 1.)]
    validation = [Observation({"u": x}, 3. * x + 2.) for x in (-.8, .2, .7)]
    parent = {"candidate": Candidate("biased_response", Program("feature", name="u"), "fit"),
              "kind": "current_affine", "family_id": "T02"}
    agent = UnifiedResearchAgent(ResearchConfig(use_memory=False))
    budget, actions = SearchBudget(2000), []

    revised = agent._refit_paid_response(spec, parent, training, validation, budget, actions)

    assert revised is not None
    assert loss(revised["candidate"].program, validation, 0., .5) < 1e-12
    assert revised["selection_basis"] == "all_acquired_evidence"
    assert revised["mechanism_revision"]["status"] == "proposed"
    assert not agent.library.concepts
    assert actions[-1]["requires_fresh_confirmation"]
    assert actions[-1]["observation_cost"] == 0
    assert actions[-1]["compute_cost"] == budget.spent > 0


def test_paid_response_refit_keeps_parent_when_heldout_evidence_disagrees():
    spec = RepeatedResponseLab.spec
    training = [Observation({"u": x}, 3. * x + 2.) for x in (-1., -.5, 0., .5, 1.)]
    validation = [Observation({"u": x}, x) for x in (-.8, .2, .7)]
    parent = {"candidate": Candidate("retained_response", Program("feature", name="u"), "fit"),
              "kind": "current_affine", "family_id": "T02"}
    agent = UnifiedResearchAgent(ResearchConfig(use_memory=False))
    budget, actions = SearchBudget(2000), []

    revised = agent._refit_paid_response(spec, parent, training, validation, budget, actions)

    assert revised is None
    assert not actions[-1]["accepted_as_working_prediction"]
    assert actions[-1]["after_validation_loss"] > actions[-1]["before_validation_loss"]
    assert budget.spent > 0
