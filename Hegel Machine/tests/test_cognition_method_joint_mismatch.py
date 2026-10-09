"""A fresh counterexample can refute a repair without beating an old error peak."""

from copy import deepcopy
from dataclasses import replace
import gzip
import importlib
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_continuation import _guard
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


@pytest.fixture(scope="module")
def joint_case():
    path = Path(__file__).parent / "fixtures/method_joint_mismatch_root10.json.gz"
    case = json.loads(gzip.decompress(path.read_bytes()))
    for batch in case["report"]["batches"]:
        history = []
        for row in batch["observations"]:
            row["history"] = list(history)
            history.append({**row["features"], "__target__": row["target"]})
    return case


@pytest.fixture(scope="module")
def joint_factory():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
        return importlib.import_module("run_observed_condition_decoding_study").laboratory


def run_case(case, factory, *, additional_rows=None):
    old = deepcopy(case["report"])
    limits = case["limits"]
    rows = limits["observation_cost"] if additional_rows is None else old["costs"]["observation_cost"] + additional_rows
    measurements = limits["measurement_cost"] if additional_rows is None else old["costs"]["measurement_cost"] + additional_rows
    config = replace(ResearchConfig(**case["config"]), seed=case["task"]["seed"],
        observation_budget=rows, compute_budget=limits["compute_cost"])
    agent = UnifiedResearchAgent(config,
        library=KnowledgeLibrary.from_dict(deepcopy(case["knowledge"])),
        measurement_budget=measurements)
    with factory(case["task"]) as lab:
        report = agent.run(lab, previous=old, continuation_candidate=deepcopy(case["candidate"]))
    return report, report["actions"][len(old["actions"]):]


def first_decision(events):
    return next(item for item in events if item["action"] == "choose_method_followup_research")


@pytest.fixture(scope="module")
def reconstructed(joint_case, joint_factory):
    return run_case(joint_case, joint_factory)


def test_actual_joint_failure_reopens_with_paid_counterexample_and_fresh_confirmation(
        joint_case, reconstructed):
    report, events = reconstructed
    decision = first_decision(events)
    assert decision["continue"]
    assert decision["reason"] == "new_revision_refuted_with_joint_response_mismatch"
    assert decision["new_max_point_loss"] < decision["previous_max_point_loss"]
    failure = next(item for item in events if item["action"] == "retain_method_confirmation_counterexample")
    assert failure["rejected_loss"] == pytest.approx(110.72188620977069)
    assert failure["retained_loss"] == pytest.approx(3.5745415304130512)
    assert failure["retained_noise_assessment"]["reason"] == "predictable_residual"
    confirmations = [item for item in events if item["action"] == "validate_continued_program"]
    assert confirmations[0]["program"] == joint_case["report"]["program"]
    assert not confirmations[0]["accepted"]
    assert len(confirmations) >= 2
    assert len({item["probe_id"] for item in confirmations}) == len(confirmations)
    assert report["batches"][:len(joint_case["report"]["batches"])] == joint_case["report"]["batches"]
    assert all(report["costs"][key] <= limit for key, limit in joint_case["limits"].items())
    assert failure["rejected_program"] in [item["program"] for item in report["continuation_state"]["alternatives"]]


def test_joint_mismatch_cannot_spend_reconstruction_without_full_confirmation(
        joint_case, joint_factory, monkeypatch):
    def unfunded_research(*args, **kwargs):
        pytest.fail("No second comparison and confirmation can be bought")

    monkeypatch.setattr(UnifiedResearchAgent, "_research_view", unfunded_research)
    report, events = run_case(joint_case, joint_factory, additional_rows=72)
    assert first_decision(events)["joint_response_mismatch"]
    stopped = next(item for item in events if item["action"] == "stop_method_response_reconstruction")
    assert stopped["reason"] == "independent_comparison_and_confirmation_unfunded"
    assert report["program"] == joint_case["report"]["program"]
    assert report["visit_costs"]["observation_cost"] == 72


def test_unchanged_reconstruction_does_not_buy_the_same_confirmation_again(
        joint_case, joint_factory, monkeypatch):
    rebuilt = []

    def unchanged(self, spec, training, validation, budget, actions, transfer, graph, *, parent, **kwargs):
        rebuilt.append((training, validation))
        return [parent]

    monkeypatch.setattr(UnifiedResearchAgent, "_research_view", unchanged)
    report, events = run_case(joint_case, joint_factory)
    assert len(rebuilt) == 1
    latest = report["batches"][-1]["observations"]
    assert [row.target for row in rebuilt[0][1]] == [row["target"] for row in latest]
    assert len(rebuilt[0][0]) + len(rebuilt[0][1]) == report["costs"]["observation_cost"]
    assert any(item.get("reason") == "no_changed_response_proposal" for item in events)
    assert report["visit_costs"]["observation_cost"] == 72
    assert report["program"] == joint_case["report"]["program"]


def test_noise_compatible_retained_response_does_not_reopen_as_joint_mismatch(
        joint_case, joint_factory, monkeypatch):
    incumbent = _guard(Program.from_dict(joint_case["report"]["program"]), joint_case["report"]["fallback"])
    original_support = UnifiedResearchAgent._support

    def calibrated_noise(self, program, *args):
        if program == incumbent:
            return {"adequate": True, "reason": "noise_compatible"}
        return original_support(self, program, *args)

    monkeypatch.setattr(UnifiedResearchAgent, "_support", calibrated_noise)
    report, events = run_case(joint_case, joint_factory)
    decision = first_decision(events)
    assert not decision["continue"] and not decision["joint_response_mismatch"]
    assert report["visit_costs"]["observation_cost"] == 72
    assert report["program"] == joint_case["report"]["program"]
    assert report["status"] == "unresolved"  # The noise check alone is not new replication.


def test_unchanged_method_failure_still_stops_without_new_response_information(
        joint_case, joint_factory, monkeypatch):
    from hegel_machine.cognition import condition_response_actions

    monkeypatch.setattr(condition_response_actions, "_observed_response_claims", lambda *args: [])
    unchanged = deepcopy(joint_case)
    unchanged["report"]["continuation_state"]["alternatives"] = []
    unchanged["candidate"].update(program=unchanged["report"]["program"],
                                  fallback=unchanged["report"]["fallback"])
    report, events = run_case(unchanged, joint_factory)
    decision = first_decision(events)
    assert not decision["continue"]
    assert decision["reason"] == "method_failed_without_new_information"
    assert not any(item["action"] == "retain_method_confirmation_counterexample" for item in events)
    assert report["program"] == joint_case["report"]["program"]
    assert not any(item["action"] == "acquire" for item in events[events.index(decision) + 1:])
    assert report["visit_costs"]["observation_cost"] == 96  # Includes the required noise replication.
