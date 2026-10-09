"""Shared research funds can reach a paid, unresolved instrument boundary."""

from copy import deepcopy
from dataclasses import replace
import gzip
import importlib
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.condition_marginal_value import CONDITION_FOLLOWUP_ACTION, condition_followup_policy
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.instrument_condition_revisits import (
    instrument_condition_offer, screen_instrument_condition_revisit,
)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget
from hegel_machine.cognition.unified_continuation import paid_view
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


@pytest.fixture(scope="module")
def paid_case():
    path = Path(__file__).parent / "fixtures/instrument_condition_revisit_root08.json.gz"
    return json.loads(gzip.decompress(path.read_bytes()))


@pytest.fixture(scope="module")
def laboratory_factory():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
        return importlib.import_module("run_observed_condition_decoding_study").laboratory


@pytest.fixture(scope="module")
def screened(paid_case, laboratory_factory):
    agent = UnifiedResearchAgent(ResearchConfig(**paid_case["config"]), measurement_budget=320)
    lab = laboratory_factory(paid_case["task"])
    budget = SearchBudget(20000)
    screen = screen_instrument_condition_revisit(agent, lab, paid_case["report"], budget)
    return dict(agent=agent, lab=lab, budget=budget, screen=screen)


POOL = dict(observation_cost=1864, measurement_cost=1984, compute_cost=2199666)


def offer_for(screened):
    remaining = {**POOL, "compute_cost": POOL["compute_cost"] - screened["budget"].spent}
    return instrument_condition_offer(screened["screen"], remaining,
        experience=screened["agent"].library.action_experience)


def test_paid_boundary_risk_can_use_shared_pool_despite_zero_point_loss(paid_case, screened):
    previous = paid_case["report"]
    assert previous["selection_loss"] < 1e-7 and previous["confirmation_loss"] < 1e-7
    assert previous["condition_uncertainty"]["expected_prediction_ambiguity"] > 7.
    assert screened["screen"] is not None
    assert 0 < screened["budget"].spent <= screened["budget"].limit
    assert screened["lab"].acquisitions == []
    offer = offer_for(screened)
    assert offer["basis"] == "unresolved_instrument_condition"
    assert offer["old_evidence_gain"] == 0.
    assert offer["predicted_response_risk_reduction"] > 0.
    assert offer["minimum_resources"]["observation_cost"] == 48
    decision = offer["investment_decision"]
    assert decision["expected_marginal_value"] > decision["resource_share"]
    assert decision["empirical_marginal_prediction_gain"] == 0.


def test_cached_screen_reprices_larger_pool_without_replanning(screened):
    screen = json.loads(json.dumps(screened["screen"]))
    before = deepcopy(screen)
    small = dict(observation_cost=116, measurement_cost=116, compute_cost=96513)
    assert instrument_condition_offer(screen, small) is None
    assert instrument_condition_offer(screen, POOL) is not None
    assert instrument_condition_offer(screen, {**POOL, "observation_cost": 47}) is None
    assert instrument_condition_offer(screen, {**POOL, "measurement_cost": 47}) is None
    assert screen == before


def test_paid_failure_history_and_real_competitors_survive_offer(paid_case, screened):
    offer = offer_for(screened)
    assert screened["screen"]["condition_research_returns"] == paid_case["report"]["condition_research_returns"]
    assert screened["screen"]["previous_experiments"]
    assert offer["candidate"]["instrument_condition_experiment"]["plan"] == screened["screen"]["plan"]
    assert all(set(point) == set(screened["lab"].spec.feature_bounds)
        for point in screened["screen"]["plan"]["probe"]["sequence"])
    context = screened["screen"]["marginal_context"]
    record = screened["agent"].library.action_experience[context][CONDITION_FOLLOWUP_ACTION]
    assert record["trials"] == 0


def test_deleted_and_recreated_context_uses_research_content_not_recycled_number(paid_case, screened):
    screen = json.loads(json.dumps(screened["screen"]))
    previous, laboratory = paid_case["report"], screened["lab"]
    channels = tuple(previous["selected_channels"])
    rows = [row for batch in paid_view(laboratory, previous, channels) for row in batch]
    history = deepcopy(screened["agent"].library.action_experience)
    history.clear()  # The study's actual deletion operation.
    empty = instrument_condition_offer(screen, POOL, experience=history)
    assert empty is not None and empty["investment_decision"]["empirical_marginal_prediction_gain"] == 0.
    unrelated = {"candidate": Candidate("other_response", Program("const", value=1.), "other")}
    replacement = condition_followup_policy(laboratory.spec_for(channels), rows, unrelated, history)
    assert replacement.action_context(CONDITION_FOLLOWUP_ACTION) == screen["marginal_context"]
    replacement.action_record(CONDITION_FOLLOWUP_ACTION).update(trials=1, expected_gain=7.)
    incompatible = instrument_condition_offer(screen, POOL, experience=history)
    assert incompatible["investment_decision"] == empty["investment_decision"]
    selected = {"candidate": Candidate("same_response", Program.from_dict(previous["program"]), "same")}
    restored = condition_followup_policy(laboratory.spec_for(channels), rows, selected, history)
    assert restored.action_context(CONDITION_FOLLOWUP_ACTION) != screen["marginal_context"]
    restored.action_record(CONDITION_FOLLOWUP_ACTION).update(trials=1, expected_gain=.8)
    equivalent = instrument_condition_offer(screen, POOL, experience=json.loads(json.dumps(history)))
    assert equivalent["investment_decision"]["empirical_marginal_prediction_gain"] == .8


@pytest.fixture(scope="module")
def executed(paid_case, laboratory_factory, screened):
    previous = paid_case["report"]
    offer = offer_for(screened)
    limits = {key: previous["costs"][key] + value
              for key, value in offer["minimum_resources"].items()}
    laboratory = laboratory_factory(paid_case["task"])
    config = replace(ResearchConfig(**paid_case["config"]), max_rounds=1,
        observation_budget=limits["observation_cost"], compute_budget=limits["compute_cost"])
    # A separate library avoids adding an executed trial to the screen fixture.
    from hegel_machine.cognition.memory import KnowledgeLibrary
    library = KnowledgeLibrary.from_dict(screened["agent"].library.to_dict())
    agent = UnifiedResearchAgent(config, measurement_budget=limits["measurement_cost"], library=library)
    report = agent.run(laboratory, previous=previous, continuation_candidate=offer["candidate"])
    return dict(report=report, offer=offer, limits=limits, library=library)


def test_funded_probe_executes_once_then_receives_new_frozen_confirmation(paid_case, executed):
    previous, report = paid_case["report"], executed["report"]
    actions = report["actions"][len(previous["actions"]):]
    acquired = [item for item in actions if item["action"] == "acquire"]
    assert [item["purpose"] for item in acquired] == [
        "distinguish_competing_observed_conditions", "confirm_continuation_program"]
    planned = executed["offer"]["candidate"]["instrument_condition_experiment"]["plan"]
    assert acquired[0]["probe_id"] == planned["probe"]["probe_id"]
    assert acquired[0]["probe_id"] != acquired[1]["probe_id"]
    assert report["batches"][:len(previous["batches"])] == previous["batches"]
    purchased = report["batches"][len(previous["batches"])]
    assert purchased["probe"]["sequence"] == planned["probe"]["sequence"]
    funding = next(item for item in actions if item["action"] == "choose_condition_research_investment")
    retained = next(item for item in actions if item["action"] == "retain_observed_condition_question")
    assert funding["investment_scope"] == "session_remaining"
    assert funding["planning_compute_cost"] == 0  # Screened probe is not planned twice.
    assert retained["compute_cost"] == 0  # Reusing its paid assessment incurs no new work.
    assert report["visit_costs"]["observation_cost"] == report["visit_costs"]["measurement_cost"] == 48
    assert report["visit_costs"]["compute_cost"] <= executed["offer"]["minimum_resources"]["compute_cost"]
    assert report["status"] == "unresolved"  # Fresh point fit has not settled the cut.


def test_only_fresh_paired_prediction_credits_the_complete_instrument_visit(paid_case, executed, screened):
    previous, report = paid_case["report"], executed["report"]
    paired, = report["continuation"]["method_marginal_comparisons"]
    batch = next(item for item in report["batches"] if item["probe"]["probe_id"] == paired["probe_id"])
    rows = [Observation(item["features"], item["target"], tuple(item["history"]))
            for item in batch["observations"]]
    assert paired["before_loss"] == pytest.approx(loss(
        Program.from_dict(previous["program"]), rows, previous["fallback"], .5))
    assert paired["after_loss"] == pytest.approx(loss(
        Program.from_dict(report["program"]), rows, report["fallback"], .5))
    assert paired["stage_costs"]["observation_cost"] == report["visit_costs"]["observation_cost"]
    assert paired["stage_costs"]["compute_cost"] == (
        report["visit_costs"]["compute_cost"] + screened["screen"]["compute_cost"])
    record = executed["library"].action_experience[paired["context"]][CONDITION_FOLLOWUP_ACTION]
    assert record["trials"] == 1
    cycle, = record["instrument_revisits"]
    assert cycle["sampling_context"] == screened["screen"]["plan"]["sampling_context"]
    assert cycle["compute_cost"] == report["visit_costs"]["compute_cost"]
    assert cycle["screening_compute_cost"] == screened["screen"]["compute_cost"]
    assert cycle["observation_cost"] == cycle["measurement_cost"] == 48


def test_restored_numeric_process_continues_independent_readings_without_rebilling(
        paid_case, executed, laboratory_factory):
    previous, report = paid_case["report"], executed["report"]
    continuous = laboratory_factory(paid_case["task"])
    from hegel_machine.cognition.models import Probe
    for batch in previous["batches"]:
        continuous.acquire(Probe(batch["probe"]["probe_id"], tuple(batch["probe"]["sequence"])), batch["channels"])
    before = continuous.costs.copy()
    for batch in report["batches"][len(previous["batches"]):]:
        actual = continuous.acquire(Probe(batch["probe"]["probe_id"],
            tuple(batch["probe"]["sequence"])), batch["channels"])
        assert [row.features for row in actual.observations] == [row["features"] for row in batch["observations"]]
        assert [row.target for row in actual.observations] == [row["target"] for row in batch["observations"]]
    for resource in ("observation_cost", "measurement_cost"):
        assert continuous.costs[resource] - before[resource] == report["visit_costs"][resource]
        assert report["costs"][resource] == previous["costs"][resource] + report["visit_costs"][resource]


def test_supported_or_real_response_failure_uses_its_existing_research_path(paid_case, laboratory_factory):
    agent = UnifiedResearchAgent(ResearchConfig(**paid_case["config"]), measurement_budget=320)
    previous = deepcopy(paid_case["report"])
    previous["condition_uncertainty"]["status"] = "supported"
    budget = SearchBudget(20000)
    assert screen_instrument_condition_revisit(agent, laboratory_factory(paid_case["task"]), previous, budget) is None
    assert budget.spent == 0
    previous["condition_uncertainty"]["status"] = "unresolved"
    previous["condition_uncertainty"]["original_target_loss"] = 1.
    assert screen_instrument_condition_revisit(agent, laboratory_factory(paid_case["task"]), previous, budget) is None
    assert budget.spent == 0


def test_funded_instrument_plan_keeps_competitors_without_unquoted_prefix_refits(
        paid_case, laboratory_factory, monkeypatch):
    from hegel_machine.cognition import unified_research

    previous = deepcopy(paid_case["report"])
    base = Program.from_dict(previous["program"])
    # Retained intercept alternatives are executable hypotheses. They should
    # receive actual paid-data comparisons, not eight unsolicited refits before
    # the already funded instrument experiment starts.
    alternatives = [{"program": Program("add", children=(base,
        Program("const", value=float(index + 1)))).to_dict(),
        "channels": previous["selected_channels"], "fallback": previous["fallback"],
        "kind": "retained_intercept_alternative", "family_id": "T03"}
        for index in range(8)]
    previous["continuation_state"]["alternatives"] = alternatives
    config = ResearchConfig(**paid_case["config"])
    agent = UnifiedResearchAgent(config, measurement_budget=320)
    laboratory = laboratory_factory(paid_case["task"])
    screen = screen_instrument_condition_revisit(agent, laboratory, previous, SearchBudget(20000))
    offer = instrument_condition_offer(screen, POOL, experience=agent.library.action_experience)
    assert offer is not None
    limits = {key: previous["costs"][key] + value for key, value in offer["minimum_resources"].items()}
    original_fit = unified_research._fit
    before_purchase = []

    def measured_fit(*args, **kwargs):
        if len(laboratory.acquisitions) <= len(previous["batches"]):
            before_purchase.append(True)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(unified_research, "_fit", measured_fit)
    report = UnifiedResearchAgent(replace(config, max_rounds=1,
        observation_budget=limits["observation_cost"], compute_budget=limits["compute_cost"]),
        measurement_budget=limits["measurement_cost"], library=agent.library).run(laboratory,
            previous=previous, continuation_candidate=offer["candidate"])
    assert not before_purchase
    events = report["actions"][len(previous["actions"]):]
    compared = next(item for item in events if item["action"] == "compare_continuation_on_paid_evidence")
    assert all(any(item["program"] == hypothesis["program"] and item["loss"] > 0.
                   for item in compared["alternatives"]) for hypothesis in alternatives)
    assert report["visit_costs"]["observation_cost"] == report["visit_costs"]["measurement_cost"] == 48
    assert report["visit_costs"]["compute_cost"] <= offer["minimum_resources"]["compute_cost"]
    acquired = [item for item in events if item["action"] == "acquire"]
    assert [item["purpose"] for item in acquired] == [
        "distinguish_competing_observed_conditions", "confirm_continuation_program"]
    assert acquired[0]["probe_id"] != acquired[1]["probe_id"]
