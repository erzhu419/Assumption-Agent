"""Fresh Agent confirmation settles source value, not just method support."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.legacy_session_feedback import confirm_source_return, settle_source_confirmation
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch, MeasurementChannel
from hegel_machine.cognition.models import Observation, Probe, SearchBudget, TaskSpec
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMAgent, InquiryLLMConfig
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from test_workflow_knowledge_returns import _setup, _consume


class SquareConfirmationLab:
    def __init__(self, sign=1.):
        self.sign, self.purchases = sign, []

    def confirmation_actions(self, count, rng):
        return [{"u": rng.uniform(-1., 1.)} for _ in range(count)]

    def execute(self, action):
        record = {"observation_index": len(self.purchases), "features": dict(action),
                  "target": self.sign * action["u"] ** 2, "history": []}
        self.purchases.append(record)
        return record

    @staticmethod
    def observation_rows(record):
        return [Observation(record["features"], record["target"], ())]


def confirmed_case(tmp_path, *, sign=1.):
    library, _, _, _ = _setup(quadratic=True)
    returned = _consume(library, quadratic=True)
    receipt = returned["returns"][0]
    assert receipt["confirmation_status"] == "provisional"
    assert receipt["final_status"] == "completed"
    selected = {"program": receipt["retained_program"], "fallback": 0.}
    code = export_temporal_program(Program.from_dict(selected["program"]))
    paid = [Observation({"u": value}, value ** 2) for value in (-1., -.5, 0., .5, 1.)]
    agent = InquiryLLMAgent(None, tmp_path, arm="code_react", source_task=False,
        config=InquiryLLMConfig(observation_budget=40, compute_budget=1000, confirmation_rows=24))
    agent.rows = list(paid)
    agent.costs["observation_cost"] = len(paid)
    agent._execute("write", {"path": "model.py", "code": code}, None)
    lab = SquareConfirmationLab(sign)
    finished = agent._execute("finish", {"model": "model.py"}, lab)
    fresh = finished["fresh_confirmation"]
    args = {"previous": {"program": Program("const", value=0.).to_dict(), "fallback": 0.},
        "selected": selected, "source_returns": returned["returns"], "paid_rows": paid,
        "fresh_rows": [row for record in fresh["observations"] for row in lab.observation_rows(record)],
        "frozen_code": code, "submitted_code": (tmp_path / "model.py").read_text(),
        "comparison_id": "agent_finish:0", "available_compute": 100,
        "passed": fresh["passed"]}
    spec = TaskSpec("consumer", "Predict a response.", ("u",), (), (),
                    feature_bounds={"u": (-1., 1.)})
    return library, spec, args, agent, lab


def test_actual_second_confirmation_promotes_source_and_updates_producing_workflow(tmp_path):
    library, spec, args, agent, lab = confirmed_case(tmp_path)
    result = settle_source_confirmation(library, spec, **args)
    agent.costs["compute_cost"] += result["compute_cost"]
    assert result["status"] == "settled" and result["confirmed_program"]
    assert len(lab.purchases) == 24 and agent.costs["observation_cost"] == 29
    assert result["compute_cost"] == 48
    assert result["after_loss"] < result["before_loss"]
    receipts = [row for row in library.concepts[0].reuse_history if row["kind"] == "research_return"]
    assert len(receipts) == 2
    assert all(row["confirmation_status"] == "confirmed" for row in receipts)
    assert all(row["confirmation_replicates"] == 2 for row in receipts)
    assert receipts[-1]["adaptation_compute"] == 0
    outcomes = library.research_workflows.methods[0]["knowledge_outcomes"]
    assert len(outcomes) == 2 and all(row["retained_gain"] > 0 for row in outcomes)
    assert all(row["source_returns"][0]["confirmation_status"] == "confirmed" for row in outcomes)
    before = deepcopy(library.to_dict())
    assert settle_source_confirmation(library, spec, **args)["reason"] == "confirmation_already_settled"
    assert library.to_dict() == before


def test_actual_failed_confirmation_keeps_prior_provisional_and_records_negative_return(tmp_path):
    library, spec, args, agent, _ = confirmed_case(tmp_path, sign=-1.)
    assert not args["passed"] and agent.model is None
    result = settle_source_confirmation(library, spec, **args)
    assert not result["confirmed_program"] and result["after_loss"] > result["before_loss"]
    receipts = [row for row in library.concepts[0].reuse_history if row["kind"] == "research_return"]
    assert receipts[0]["confirmation_status"] == "provisional"
    assert receipts[-1]["confirmation_status"] == "rejected"
    assert receipts[-1]["retained_normalized_gain"] < 0
    assert library.research_workflows.methods[0]["knowledge_outcomes"][-1]["retained_gain"] < 0


@pytest.mark.parametrize("change,reason", [
    ("code", "submitted_model_differs_from_frozen_source_program"),
    ("budget", "insufficient_paired_comparison_compute"),
    ("measurement", "source_or_incumbent_requires_unavailable_prediction_fields"),
])
def test_unavailable_confirmation_leaves_existing_receipts_unchanged(tmp_path, change, reason):
    library, spec, args, _, _ = confirmed_case(tmp_path)
    if change == "code":
        args["submitted_code"] += "\n# model was subsequently edited\n"
    elif change == "budget":
        args["available_compute"] = 47
    else:
        args["selected"]["program"] = Program("feature", name="meter").to_dict()
    before = deepcopy(library.to_dict())
    assert settle_source_confirmation(library, spec, **args) == {
        "status": "deferred", "reason": reason, "compute_cost": 0}
    assert library.to_dict() == before


class MeasuredSquareLab:
    channels = (MeasurementChannel("meter", 3, "A purchased auxiliary reading."),)

    def __init__(self, sign=1.):
        self.sign, self.purchases = sign, []

    def spec_for(self, channels=()):
        return TaskSpec("measured_consumer", "Predict the response.", ("u", *channels), (), (),
                        feature_bounds={"u": (-1., 1.)})

    def acquire(self, probe, channels=()):
        rows = []
        for point in probe.sequence:
            meter = .5 * point["u"] + .2
            features = {**point, **({"meter": meter} if "meter" in channels else {})}
            rows.append(Observation(features, self.sign * meter ** 2))
        batch = MeasurementBatch(rows, len(rows), 3 * len(rows) if "meter" in channels else 0,
                                 tuple(channels))
        self.purchases.append(batch)
        return batch


def measured_case(*, sign=1.):
    library, _, _, _ = _setup(quadratic=True)
    lab = MeasuredSquareLab()
    spec = lab.spec_for(("meter",))
    paid = lab.acquire(Probe("paid", tuple({"u": value} for value in (-1., -.5, 0., .5, 1.))),
                       ("meter",)).observations
    first = lab.acquire(Probe("first_confirmation", tuple({"u": value} for value in (-.8, -.3, .3, .8))),
                        ("meter",)).observations
    library.begin_transfer_episode(spec, paid)
    budget = SearchBudget(10000)
    offers = library.propose(spec, paid, budget, preserve_parameters=False)
    winner = min(offers, key=lambda item: loss(item.program, first, 0., .5))
    assert ("meter", 0) in winner.program.observed_variables()
    zero = Program("const", value=0.)
    after, before = loss(winner.program, first, 0., .5), loss(zero, first, 0., .5)
    library.record_transfer_result(winner, spec, accepted=True, loss=after, baseline_loss=before,
        evidence_rows=len(first), comparison_compute=2 * len(first), comparison_id="first_confirmation")
    result = library.finish_transfer_episode(spec, final_candidate=winner, final_loss=after,
        baseline_loss=before, compute_cost=budget.spent + 2 * len(first),
        observation_cost=len(first), validation_rows=len(first), final_status="completed")
    assert result["returns"][0]["confirmation_status"] == "provisional"
    lab.sign = sign
    args = {"previous": {"program": zero.to_dict(), "fallback": 0., "selected_channels": []},
        "selected": {"program": winner.program.to_dict(), "fallback": 0., "selected_channels": ["meter"]},
        "source_returns": result["returns"], "paid_rows": paid,
        "probe": Probe("second_confirmation", tuple({"u": value} for value in (-.7, -.1, .2, .9))),
        "available_compute": 8, "observation_allowance": 4, "measurement_allowance": 12,
        "comparison_id": "second_confirmation"}
    return library, lab, args


@pytest.mark.parametrize("sign,expected", [(1., "confirmed"), (-1., "rejected")])
def test_active_confirmation_buys_real_meter_values_and_settles_existing_returns(sign, expected):
    library, lab, args = measured_case(sign=sign)
    with pytest.raises(ValueError, match="only control"):
        export_temporal_program(Program.from_dict(args["selected"]["program"]))
    purchases = len(lab.purchases)
    result = confirm_source_return(library, lab, **args)
    assert len(lab.purchases) == purchases + 1
    assert result["new_costs"] == {"observation_cost": 4, "measurement_cost": 12,
                                   "compute_cost": 8, "queries": 1}
    assert result["channels"] == ["meter"]
    assert all("meter" in row["features"] for row in result["batch"]["observations"])
    assert result["knowledge_return"]["returns"][0]["confirmation_status"] == expected
    receipts = [row for row in library.concepts[0].reuse_history if row["kind"] == "research_return"]
    assert receipts[0]["confirmation_status"] == ("confirmed" if sign == 1. else "provisional")
    assert result["knowledge_return"]["method_returns"]
    # Settlement does not change the native program into a control-only export.
    with pytest.raises(ValueError, match="only control"):
        export_temporal_program(Program.from_dict(args["selected"]["program"]))


def test_active_confirmation_checks_complete_cost_before_buying_measurements():
    library, lab, args = measured_case()
    args["measurement_allowance"] = 11
    purchases = len(lab.purchases)
    before = deepcopy(library.to_dict())
    result = confirm_source_return(library, lab, **args)
    assert result["status"] == "deferred" and len(lab.purchases) == purchases
    assert library.to_dict() == before
