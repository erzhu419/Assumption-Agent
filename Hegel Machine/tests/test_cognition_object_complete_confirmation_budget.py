"""An object-response revision must return a parent and fund its confirmation."""

from dataclasses import asdict, replace

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig
from hegel_machine.cognition.object_research_actions import ObjectResearchActions
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


class ConstantObjectResponseLab(UnifiedObjectLab):
    """Keep the ordinary object instruments/costs, with a known test response."""

    def __init__(self):
        super().__init__(objects=2, readout="total", seed=71)
        self.received_rows = {}

    def acquire(self, probe, channels=()):
        batch = super().acquire(probe, channels)
        history, rows = [], []
        for row in batch.observations:
            rows.append(Observation(row.features, 1., tuple(history)))
            history.append({**row.features, "__target__": 1.})
        batch.observations = rows
        self.received_rows[probe.probe_id] = rows
        return batch


def paid_object_scene(laboratory):
    probe = laboratory.random_probe(29, length=12, name="already_paid")
    batch = laboratory.acquire(probe, ("fine",))
    return {"status": "unresolved", "costs": {**laboratory.costs, "compute_cost": 24},
        "actions": [], "batches": [{"probe": asdict(probe), "channels": ["fine"],
            "observations": [asdict(row) for row in batch.observations],
            "transitions": batch.transitions}],
        "program": Program("const", value=0.).to_dict(), "selected_channels": ["fine"],
        "selected_kind": "current_affine", "fallback": 0.,
        "selection_loss": 1., "confirmation_loss": None}


def object_agent(*, observations=192, compute=12000, measurement=1200):
    config = ResearchConfig(observation_budget=observations, compute_budget=compute,
        max_rounds=2, use_memory=True, active_probes=False, risk_learning=True,
        minimum_improvement=1e-7, learn_research_policy=False)
    return UnifiedResearchAgent(config, measurement_budget=measurement)


@pytest.mark.parametrize("compute", [12000, 130000])
def test_native_object_consumes_paid_answer_without_refit_and_confirms_after_search_exhausts(
        monkeypatch, compute):
    laboratory = ConstantObjectResponseLab()
    previous = paid_object_scene(laboratory)
    agent = object_agent(compute=compute, measurement=(12 + 24 + 48) * 6)
    support_calls = []

    def saved_answer(program, training, validation, spec, budget):
        support_calls.append({"limit": budget.limit, "training": list(training),
                              "validation": list(validation)})
        assert budget.spend(budget.remaining)
        return {"adequate": False, "reason": "structured_counterexample",
                "structural_counterexample": {"kind": "saved_structure", "program":
                    Program("const", value=1.).to_dict()},
                "counterexample_baseline_program": program.to_dict()}

    def repeated_fit(*args, **kwargs):
        pytest.fail("The already paid residual answer must be consumed before another fit")

    monkeypatch.setattr(agent, "_support", saved_answer)
    monkeypatch.setattr(agent, "_research_view", repeated_fit)

    report = agent.run(laboratory, previous=previous)

    assert report["status"] == "completed"
    assert len(support_calls) == 1
    assert 0 < support_calls[0]["limit"] <= 90000
    actions = report["actions"]
    consumed, = [event for event in actions if event["action"] == "consume_paid_residual_answer"]
    assert consumed["status"] == "improved"
    assert consumed["new_fits"] == consumed["new_observations"] == 0
    assert consumed["requires_fresh_confirmation"]
    validations = [event for event in actions if event["action"] == "validate_frozen_program"]
    assert [event["accepted"] for event in validations] == [False, True]
    assert [event["observation_rows"] for event in validations] == [24, 48]
    completed_at = next(index for index, event in enumerate(actions)
                        if event["action"] == "validate_frozen_program" and event["accepted"])
    # Optional deployment calibration after this acceptance is a separate use.
    purchases = [event for event in actions[:completed_at] if event["action"] == "acquire"]
    assert [event["purpose"] for event in purchases] == ["confirm_frozen_program"] * 2
    last_rows = laboratory.received_rows[purchases[-1]["probe_id"]]
    support_rows = support_calls[0]["training"] + support_calls[0]["validation"]
    assert not {id(row) for row in last_rows} & {id(row) for row in support_rows}
    assert sum(event["observation_cost"] for event in purchases) == 72
    assert report["costs"]["compute_cost"] <= compute
    assert report["batches"][0] == previous["batches"][0]


@pytest.mark.parametrize("compute,measurement,answer_retained", [
    (12000, 384, True), (200, 1200, False)])
def test_native_object_unfunded_next_confirmation_does_not_purchase_or_refit(
        monkeypatch, compute, measurement, answer_retained):
    laboratory = ConstantObjectResponseLab()
    previous = paid_object_scene(laboratory)
    # Initial 24-row confirmation fits; the required next 48-row one does not.
    agent = object_agent(compute=compute, measurement=measurement)

    def saved_answer(program, training, validation, spec, budget):
        return {"adequate": False, "reason": "structured_counterexample",
                "structural_counterexample": {"kind": "saved_structure", "program":
                    Program("const", value=1.).to_dict()},
                "counterexample_baseline_program": program.to_dict()}

    def repeated_fit(*args, **kwargs):
        pytest.fail("An unaffordable complete follow-up must not start another fit")

    monkeypatch.setattr(agent, "_support", saved_answer)
    monkeypatch.setattr(agent, "_research_view", repeated_fit)
    report = agent.run(laboratory, previous=previous)

    assert report["status"] == "unresolved"
    assert report["program"] == previous["program"]
    # The frozen tested incumbent remains the reported model; the unconfirmed
    # improvement stays in the saved pool for a separately funded continuation.
    assert any(entry["kind"] == "noise_residual_answer" and entry["validation_loss"] == 0.
               for entry in report["object_continuation_state"]["finalists"]) == answer_retained
    assert any(event["action"] == "defer_confirmation_revision"
               and event["reason"] == "unfunded_object_revision_and_confirmation"
               for event in report["actions"])
    confirmations = [event for event in report["actions"]
                     if event["action"] == "validate_frozen_program"]
    assert len(confirmations) == 1 and not confirmations[0]["accepted"]
    assert len(laboratory.acquisitions) == 2  # original paid scene, one failed confirmation


def test_native_object_retry_reserves_next_confirmation_without_changing_experiment(monkeypatch):
    laboratory = ConstantObjectResponseLab()
    previous = paid_object_scene(laboratory)
    agent = object_agent(measurement=648)
    agent.config = replace(agent.config, risk_learning=False)
    fitted, retries = [], []
    original = ObjectResearchActions._structure_view

    def record_retry(controller, *args, **kwargs):
        retries.append({"length": controller.length,
                        "confirmation_length": controller.confirmation_length,
                        "reserve": controller.budget.reserve})
        return original(controller, *args, **kwargs)

    def consume_fit(spec, training, validation, budget, actions, transfer, graph, **kwargs):
        program = Program("const", value=1.)
        fitted.append({"training": list(training), "validation": list(validation),
                       "limit": budget.limit})
        assert budget.spend(budget.remaining)
        return [{"candidate": Candidate("new_object_response", program, "fit"),
                 "kind": "current_affine", "family_id": "T02",
                 "training_loss": loss(program, training, 0., .5),
                 "validation_loss": loss(program, validation, 0., .5)}]

    monkeypatch.setattr(ObjectResearchActions, "_structure_view", record_retry)
    monkeypatch.setattr(agent, "_research_view", consume_fit)
    report = agent.run(laboratory, previous=previous)

    assert report["status"] == "completed"
    assert len(fitted) == len(retries) == 1
    assert retries[0]["length"] == 24
    assert retries[0]["confirmation_length"] == 48
    assert retries[0]["reserve"] >= 3 * 48
    assert len(fitted[0]["validation"]) == 24
    confirmations = [event for event in report["actions"]
                     if event["action"] == "validate_frozen_program"]
    assert [(event["observation_rows"], event["accepted"]) for event in confirmations] == [
        (24, False), (48, True)]
    feedback = [event for event in report["actions"]
                if event["action"] == "object_research_feedback" and event["confirmed"]]
    assert feedback and feedback[0]["before_loss"] == 1.
    assert feedback[0]["after_loss"] == 0.
    assert report["costs"]["compute_cost"] <= agent.config.compute_budget
    assert report["costs"]["measurement_cost"] <= agent.measurement_budget
    assert not any(event["action"] == "defer_frozen_confirmation" for event in report["actions"])
