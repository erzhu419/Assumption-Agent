"""The temporal interface reuses the original measured experiment-method loop."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.instrument_method_lab import InstrumentMethodLab
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent


BOOK = "shared/instrument_methods.json"
INSTRUMENT = "instrument.json"


def agent_for(workspace, knowledge=None, *, arm="hegel"):
    return TemporalOperatorAgent(None, workspace, arm=arm, source_task=False,
        knowledge=json.loads(json.dumps(knowledge)) if knowledge else None,
        config=InquiryLLMConfig(observation_budget=120, compute_budget=300000,
                               execution_seconds=30.))


def laboratory(stage, mode="recurrent"):
    # Only the public-action adapter is reused here. The physical apparatus is
    # the original v0.92 fixture, with its original three-device sequence.
    lab = TemporalInquiryLab(task_index=1)
    lab._laboratory = InstrumentMethodLab(stage, 157, 739, mode)
    return lab


def expand_interface(lab, stage, mode="recurrent"):
    # Adjacent stages are the same apparatus; only the legal horizon expands.
    # Preserve already paid global row indices when swapping its interface.
    lab._past_costs = lab.costs
    lab._laboratory = InstrumentMethodLab(stage, 157, 739, mode)


def operator(agent, lab, name, **arguments):
    return agent._execute("run", {"operator": name, **arguments}, lab)


def fit_short(agent, lab, length):
    rng = Random(157)
    actions = [{"kind": "trajectory", "sequence": [
        {"u": rng.uniform(-1., 1.)} for _ in range(length)], "channels": ["meter"]}
        for _ in range(7)]
    bought = agent._execute("query", {"actions": actions[:-1]}, lab)
    assert len(bought["record_indices"]) == 6
    model = operator(agent, lab, "instrument", path=INSTRUMENT)
    assert model["status"] == "requires_fresh_validation"
    validation = agent._execute("query", {"actions": actions[-1:]}, lab)
    assessed = operator(agent, lab, "assess_instrument", instrument=INSTRUMENT,
                        records=validation["record_indices"])
    assert assessed["status"] == "ambiguous_on_measured_trajectory"
    return assessed


def methods(agent):
    path = agent.workspace / BOOK
    return json.loads(path.read_text()) if path.exists() else []


def propose(agent, lab):
    return operator(agent, lab, "disagreement", instrument=INSTRUMENT)


def acquire_and_assess(agent, lab, proposal, action=None):
    return agent._execute("query", {
        "actions": [proposal["action"] if action is None else action],
        "after_observation": [{"operator": "assess_instrument",
            "instrument": INSTRUMENT, "experiment": proposal["experiment"],
            "records": "acquired"}]}, lab)


@pytest.mark.parametrize("mode", ["finite", "recurrent"])
def test_original_three_apparatus_method_is_learned_extended_and_reused(tmp_path, mode):
    first, lab = agent_for(tmp_path / "first"), laboratory(0, mode)
    fit_short(first, lab, 3)
    expand_interface(lab, 1, mode)
    before = deepcopy(lab.costs)
    initial = propose(first, lab)
    assert len(initial["action"]["sequence"]) == 4
    assert lab.costs == before and methods(first) == []
    acquired = acquire_and_assess(first, lab, initial)
    assessed = acquired["after_observation_results"][0]
    assert assessed["status"] == "supported_on_measured_trajectory"
    assert assessed["method_feedback"]["action"] == "learn_instrument_experiment_method"
    original = methods(first)[0]
    assert len(original["support"]) == 1
    assert original["support"][0]["observation_rows"] == 4
    assert original["support"][0]["measurement_cost"] == 12

    outgoing = first.result("researching")["outgoing_knowledge"]
    assert BOOK in outgoing["files"]
    second, lab = agent_for(tmp_path / "second", outgoing), laboratory(2, mode)
    assert BOOK in second.inherited_files
    fit_short(second, lab, 4)
    before_methods, before_costs = deepcopy(methods(second)), deepcopy(lab.costs)
    capped = propose(second, lab)
    assert capped["action"] is None
    assert methods(second) == before_methods and lab.costs == before_costs
    expand_interface(lab, 3, mode)
    revised = propose(second, lab)
    assert revised["reason"] == "revised_instrument_experiment_method"
    assert revised["parent_method_id"] == original["method_id"]
    assert len(revised["action"]["sequence"]) == 5
    assert revised["action"]["sequence"][:4] == initial["action"]["sequence"]
    acquired = acquire_and_assess(second, lab, revised)
    assessed = acquired["after_observation_results"][0]
    assert assessed["status"] == "supported_on_measured_trajectory"
    assert assessed["method_feedback"]["parent_method_id"] == original["method_id"]
    old, extension = methods(second)
    assert len(old["support"]) == len(extension["support"]) == 1
    assert extension["instrument_recipe"]["length"] == 5
    assert extension["support"][0]["measurement_cost"] == 15

    third, lab = agent_for(tmp_path / "third", second.result("researching")[
        "outgoing_knowledge"]), laboratory(4, mode)
    fit_short(third, lab, 4)
    expand_interface(lab, 5, mode)
    reused = propose(third, lab)
    assert reused["reason"] == "learned_instrument_experiment_method"
    assert reused["method_id"] == extension["method_id"]
    assert reused["action"] == revised["action"]
    # Procedure transfer keeps the preparation, but forecasts new fitted laws.
    assert reused["predictions"] != revised["predictions"]
    acquired = acquire_and_assess(third, lab, reused)
    assessed = acquired["after_observation_results"][0]
    assert assessed["status"] == "supported_on_measured_trajectory"
    assert assessed["method_feedback"]["action"] == "use_instrument_experiment_method"
    assert assessed["method_feedback"]["cross_task"]
    assert len(methods(third)) == 2 and len(methods(third)[1]["support"]) == 2
    assert all(agent.costs["model_calls"] == 0 for agent in (first, second, third))


@pytest.mark.parametrize("arm", ["code_react", "textual_memory", "hegel"])
def test_methods_are_ordinary_shared_files_and_one_purchase_is_counted_once(tmp_path, arm):
    agent, lab = agent_for(tmp_path, arm=arm), laboratory(0)
    fit_short(agent, lab, 3)
    expand_interface(lab, 1)
    proposal = propose(agent, lab)
    acquired = acquire_and_assess(agent, lab, proposal)
    assert acquired["after_observation_results"][0]["method_feedback"]["action"] == (
        "learn_instrument_experiment_method")
    before = deepcopy(methods(agent))
    assert len(before) == 1 and len(before[0]["support"]) == 1
    with pytest.raises(ValueError, match="already has recorded method feedback"):
        operator(agent, lab, "assess_instrument", instrument=INSTRUMENT,
            records=acquired["record_indices"], experiment=proposal["experiment"])
    assert methods(agent) == before
    # Existing arm semantics determine what crosses tasks; the measurement and
    # method operators themselves are available identically to all three arms.
    outgoing = agent.result("researching")["outgoing_knowledge"]
    assert (BOOK in outgoing.get("files", {})) == (arm == "hegel")


def test_planned_or_unmeasured_experiment_does_not_support_a_method(tmp_path):
    agent, lab = agent_for(tmp_path), laboratory(0)
    fit_short(agent, lab, 3)
    expand_interface(lab, 1)
    proposal = propose(agent, lab)
    assert methods(agent) == []
    # The interface permits buying responses without buying the meter. Such a
    # query cannot support an instrument experiment even at the planned inputs.
    unmeasured = {**deepcopy(proposal["action"]), "channels": []}
    acquired = acquire_and_assess(agent, lab, proposal, unmeasured)
    assert not any(method["support"] for method in methods(agent))
    assert acquired.get("after_observation_error") or not acquired[
        "after_observation_results"][0].get("method_feedback")


def test_different_paid_trajectory_cannot_support_the_named_experiment(tmp_path):
    agent, lab = agent_for(tmp_path), laboratory(0)
    fit_short(agent, lab, 3)
    expand_interface(lab, 1)
    proposal = propose(agent, lab)
    # This other real meter purchase may assess hypotheses in its own right;
    # its outcome cannot be recorded as evidence for the proposed preparation.
    other = deepcopy(proposal["action"])
    other["sequence"] = [{"u": .37} for _ in other["sequence"]]
    assert other != proposal["action"]
    acquired = acquire_and_assess(agent, lab, proposal, other)
    assert not any(method["support"] for method in methods(agent))
    assert acquired.get("after_observation_error") or not acquired[
        "after_observation_results"][0].get("method_feedback")
