"""Actual attempted responses own their measured gains; routes own their costs."""

from copy import deepcopy
from dataclasses import replace
import gzip
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, TaskSpec
from hegel_machine.cognition.numeric_method_revisits import (
    METHOD_RESPONSE_ACTION, METHOD_VISIT_ACTION, method_response_applicability,
    record_numeric_method_visit, refresh_numeric_method_quote,
)
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from test_cognition_method_feedback_session import laboratory_factory, restore_histories
from test_cognition_method_joint_mismatch import joint_case, joint_factory
from test_cognition_method_revisit_opportunity import real_method_stream


FIXTURES = Path(__file__).parent / "fixtures"
RESOURCES = ("observation_cost", "measurement_cost", "compute_cost")


def read_fixture(name):
    return json.loads(gzip.decompress((FIXTURES / name).read_bytes()))


@pytest.fixture(scope="module")
def executed_visit(joint_case, joint_factory):
    """Current execution, without requiring the historical failure to recur."""
    previous = deepcopy(joint_case["report"])
    config = replace(ResearchConfig(**joint_case["config"]),
        seed=joint_case["task"]["seed"],
        observation_budget=previous["costs"]["observation_cost"] + 72,
        compute_budget=joint_case["limits"]["compute_cost"])
    agent = UnifiedResearchAgent(config,
        library=KnowledgeLibrary.from_dict(deepcopy(joint_case["knowledge"])),
        measurement_budget=previous["costs"]["measurement_cost"] + 72)
    candidate = deepcopy(joint_case["candidate"])
    with joint_factory(joint_case["task"]) as lab:
        spec = lab.spec_for(candidate["channels"])
        report = agent.run(lab, previous=previous, continuation_candidate=candidate)
        before_credit = deepcopy(agent.library.to_dict())
        record_numeric_method_visit(agent.library, spec, previous, report,
            candidate["method_opportunity"], candidate["method_opportunity"]["planning_compute_cost"])
    return dict(previous=previous, candidate=candidate, spec=spec, report=report,
                before_credit=before_credit, knowledge=deepcopy(agent.library.to_dict()))


@pytest.fixture(scope="module")
def failed_visit(joint_case):
    """Score the recorded failed program; never relabel today's successful run."""
    actual = read_fixture("draft_binding_counterexample_root10.json.gz")
    recorded = read_fixture("method_response_failure_root10.json.gz")
    entry = actual["entry"]
    pair = deepcopy(recorded["comparison"])
    assert recorded["rejected_program"] == entry["program"]
    assert actual["probe_id"] == pair["probe_id"]
    rows, history = [], []
    for point in recorded["fresh_rows"]:
        rows.append(Observation(point["features"], point["target"], tuple(history)))
        history.append({**point["features"], "__target__": point["target"]})
    assert loss(Program.from_dict(entry["program"]), rows, entry["fallback"], .5) == pytest.approx(
        pair["attempted_after_loss"])
    assert loss(Program.from_dict(pair["baseline_program"]), rows,
        pair["baseline_fallback"], .5) == pytest.approx(pair["before_loss"])
    spec = TaskSpec(**actual["spec"])
    pair.update(attempted_program=deepcopy(entry["program"]),
        attempted_fallback=entry["fallback"], attempted_channels=list(entry["channels"]),
        attempted_source_draft_ids=[entry["research_draft_id"]],
        attempted_method_applicability=method_response_applicability(
            Program.from_dict(entry["program"]), spec, {"candidate": Candidate("frozen_before",
                Program.from_dict(pair["baseline_program"]), "retained_incumbent")}))
    previous, candidate = deepcopy(joint_case["report"]), deepcopy(joint_case["candidate"])
    report = {"program": deepcopy(previous["program"]), "status": "unresolved",
        "visit_costs": recorded["visit_costs"], "continuation": {"method_fresh_comparisons": [pair]}}
    library = KnowledgeLibrary.from_dict(deepcopy(joint_case["knowledge"]))
    library.research_drafts.append(deepcopy(actual["draft"]))
    before_credit = deepcopy(library.to_dict())
    record_numeric_method_visit(library, spec, previous, report,
        candidate["method_opportunity"], candidate["method_opportunity"]["planning_compute_cost"])
    return dict(previous=previous, candidate=candidate, spec=spec, report=report,
                before_credit=before_credit, knowledge=deepcopy(library.to_dict()))


def response_record(case):
    outcome = case["report"]["continuation"]["method_visit_outcome"]
    confirmation, = outcome["response_confirmations"]
    record = case["knowledge"]["action_experience"][confirmation["method_context"]][METHOD_RESPONSE_ACTION]
    return outcome, confirmation, record


def test_actual_failed_predictor_has_negative_gain_despite_route_fallback(failed_visit):
    case = failed_visit
    pair, = case["report"]["continuation"]["method_fresh_comparisons"]
    assert case["candidate"]["method_opportunity"]["draft_ids"] == ["research_draft_0006"]
    assert pair["attempted_source_draft_ids"] == ["research_draft_0011"]
    assert pair["before_loss"] == pair["after_loss"] == pytest.approx(3.5745415304130512)
    assert pair["attempted_after_loss"] == pytest.approx(110.72188620977069)
    expected = method_response_applicability(Program.from_dict(pair["attempted_program"]),
        case["spec"], {"candidate": Candidate("frozen_before",
            Program.from_dict(pair["baseline_program"]), "retained_incumbent")})
    assert pair["attempted_method_applicability"] == expected
    outcome, confirmation, record = response_record(case)
    assert record["knowledge_applicability"] == expected
    assert record["last_before_loss"] == pair["before_loss"]
    assert record["last_after_loss"] == pair["attempted_after_loss"]
    assert record["last_gain"] == record["expected_gain"] == -1.
    assert confirmation["source_draft_ids"] == ["research_draft_0011"]
    route = case["knowledge"]["action_experience"][outcome["method_context"]][METHOD_VISIT_ACTION]
    assert route["last_gain"] == 0.
    assert case["report"]["program"] == case["previous"]["program"]
    for resource in RESOURCES:
        assert record[f"mean_{resource}"] == 0.
        assert route["execution_costs"][resource] == case["report"]["visit_costs"][resource]
    assert route["execution_costs"]["visits"] == 1
    assert outcome["whole_visit_costs"]["compute_cost"] == (
        case["report"]["visit_costs"]["compute_cost"] + outcome["screening_compute_cost"])


def test_only_actual_draft_receives_response_outcome_after_json_restore(failed_visit):
    case = failed_visit
    before = {item["draft_id"]: item for item in case["before_credit"]["research_drafts"]}
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(case["knowledge"])))
    changed = {}
    for draft in library.research_drafts:
        old = before[draft["draft_id"]].get("outcomes", [])
        extra = draft.get("outcomes", [])[len(old):]
        if extra:
            changed[draft["draft_id"]] = extra
    assert set(changed) == {"research_draft_0011"}
    evidence, = changed["research_draft_0011"]
    assert evidence["kind"] == "method_response_confirmation"
    assert evidence["after_loss"] == pytest.approx(110.72188620977069)
    assert evidence["retained_loss"] == pytest.approx(3.5745415304130512)
    assert evidence["gain"] == -1.
    pair, = case["report"]["continuation"]["method_fresh_comparisons"]
    assert evidence["program"] == pair["attempted_program"]
    assert evidence["fallback"] == pair["attempted_fallback"]
    assert evidence["channels"] == pair["attempted_channels"]


def test_failed_actual_basis_reprices_its_paid_fit_without_relabeling_route_gain(failed_visit):
    """Static replay of the paid fit quote, not a claim of a historical offer."""
    case = failed_visit
    pair, = case["report"]["continuation"]["method_fresh_comparisons"]
    rows = [Observation(point["features"], point["target"], tuple(point["history"]))
            for batch in case["previous"]["batches"] for point in batch["observations"]]
    baseline = loss(Program.from_dict(pair["baseline_program"]), rows, pair["baseline_fallback"], .5)
    attempted = loss(Program.from_dict(pair["attempted_program"]), rows, pair["attempted_fallback"], .5)
    assert attempted < baseline  # The real failure had fitted this paid prefix.
    candidate = deepcopy(case["candidate"])
    candidate.update(program=pair["attempted_program"], fallback=pair["attempted_fallback"],
        research_draft_id=pair["attempted_source_draft_ids"][0])
    opportunity = candidate["method_opportunity"]
    opportunity.update(paid_counterexample_loss=baseline, candidate_paid_loss=attempted,
        method_applicability=pair["attempted_method_applicability"])
    outcome, confirmation, _ = response_record(case)
    # Requote this actual fitted response with its purchased legal probe and
    # the existing complete one-proposal method allocation, without executing it.
    length = len(candidate["method_experiment"]["probe"]["sequence"])
    quote = {"candidate": candidate, "old_evidence_gain": baseline - attempted,
        "pricing_inputs": {"method_context": confirmation["method_context"].replace(
            f":{METHOD_RESPONSE_ACTION}:", f":{METHOD_VISIT_ACTION}:"),
            "observation_quote": length + 48,
            "compute_quote": opportunity["duration_compute_quote"]
                + opportunity["response_compute_quote"] + 7 * len(rows) + 32 * 24,
            "task_compute_cap": 320000, "measurement_rate": 1., "batch_length": 24,
            "condition_uncertainty": None, "condition_research_returns": None, "tolerance": 1e-7}}
    resources = {"observation_cost": 320, "measurement_cost": 320, "compute_cost": 640000}
    cold = deepcopy(quote)
    assert refresh_numeric_method_quote(cold, {}, resources) is cold
    assert cold["expected_marginal_value"] > 0.
    assert refresh_numeric_method_quote(quote, case["knowledge"]["action_experience"], resources) is None
    assert quote["method_feedback"]["expected_gain"] == -1.
    assert quote["method_feedback"]["trials"] == 1
    assert quote["expected_marginal_value"] == 0.


def test_current_producer_records_the_executed_predictor_not_the_screened_source(executed_visit):
    case = executed_visit
    pair, = case["report"]["continuation"]["method_fresh_comparisons"]
    assert case["candidate"]["method_opportunity"]["draft_ids"] == ["research_draft_0006"]
    assert pair["attempted_source_draft_ids"] == ["research_draft_0011"]
    assert pair["attempted_after_loss"] < pair["before_loss"]
    expected = method_response_applicability(Program.from_dict(pair["attempted_program"]),
        case["spec"], {"candidate": Candidate("frozen_before",
            Program.from_dict(pair["baseline_program"]), "retained_incumbent")})
    outcome, confirmation, record = response_record(case)
    assert pair["attempted_method_applicability"] == expected
    assert record["knowledge_applicability"] == expected
    assert confirmation["gain"] > 0.
    before = {item["draft_id"]: len(item.get("outcomes", []))
              for item in case["before_credit"]["research_drafts"]}
    additions = {item["draft_id"]: item.get("outcomes", [])[before[item["draft_id"]]:]
        for item in case["knowledge"]["research_drafts"]
        if len(item.get("outcomes", [])) > before[item["draft_id"]]}
    assert set(additions) == {"research_draft_0011"}
    assert additions["research_draft_0011"][0]["program"] == pair["attempted_program"]
    assert outcome not in additions["research_draft_0011"]


@pytest.fixture(scope="module")
def root11_quote(laboratory_factory):
    case = read_fixture("method_feedback_root11.json.gz")
    state = deepcopy(case["state"])
    restore_histories(state)
    session = UnifiedResearchSession(state=state, lab_factory=laboratory_factory)
    session._offer_retained_methods(case["target_position"])
    attempt, = session.state["method_revisit_attempts"]
    with laboratory_factory(case["state"]["tasks"][0]) as lab:
        spec = lab.spec_for(attempt["cached_quote"]["candidate"]["channels"])
    return dict(quote=deepcopy(attempt["cached_quote"]), case=case,
                remaining=session._remaining(), spec=spec)


def execution_cost_fixture(quote):
    """Accounting fixture: real ledger, synthetic matching scope, no gain label.

    This tests the cost formula, not applicability of root11's old recipe to a
    newly fitted response. Current producer tests separately check real scope.
    """
    paid = read_fixture("method_execution_cost_root11.json.gz")
    context = quote["pricing_inputs"]["method_context"]
    record = {"knowledge_applicability": deepcopy(
        quote["candidate"]["method_opportunity"]["method_applicability"]),
        "execution_costs": {"visits": 1,
            **{key: paid["execution_costs"][key] for key in RESOURCES}}}
    return paid, {context: {METHOD_VISIT_ACTION: record}}


def test_recorded_route_success_does_not_become_response_value(root11_quote):
    quote = deepcopy(root11_quote["quote"])
    paid = read_fixture("method_execution_cost_root11.json.gz")
    experience = deepcopy(root11_quote["case"]["actual_method_feedback"])
    original = deepcopy(experience)
    pair = paid["fresh_comparison"]
    actual = method_response_applicability(Program.from_dict(pair["program"]), root11_quote["spec"],
        {"candidate": Candidate("actual_historical_baseline",
            Program.from_dict(pair["baseline_program"]), "retained_incumbent")})
    # This historical route began with a window-5 method but its successful
    # response is window-7; those gains cannot validate the old cached law.
    assert actual != quote["candidate"]["method_opportunity"]["method_applicability"]
    priced = refresh_numeric_method_quote(quote, experience, root11_quote["remaining"])
    assert quote["old_evidence_gain"] < 0.
    assert quote["candidate"]["method_opportunity"]["predicted_separation"] == 0.
    # Its old recipe scope differs too; neither relabel the route return nor
    # assign its cost history to this new fitted predictor by assumption.
    assert quote["method_feedback"]["route_trials"] == 0
    assert quote["method_feedback"]["trials"] == 0
    assert quote["method_feedback"]["expected_gain"] is None
    assert quote["expected_marginal_value"] == 0. and priced is None
    assert experience == original


def test_recorded_execution_cost_ignores_this_opportunity_screen_fee(root11_quote):
    original = root11_quote["quote"]
    paid, experience = execution_cost_fixture(original)
    historical_screen = paid["screening_compute_cost"]
    current_screen = original["candidate"]["method_opportunity"]["planning_compute_cost"]
    assert historical_screen != current_screen
    assert paid["whole_visit_costs"]["compute_cost"] == (
        paid["execution_costs"]["compute_cost"] + historical_screen)
    quotes = []
    for fee in (historical_screen, current_screen):
        quote = deepcopy(original)
        quote["candidate"]["method_opportunity"]["planning_compute_cost"] = fee
        refresh_numeric_method_quote(quote, experience, root11_quote["remaining"])
        quotes.append(quote)
    assert quotes[0]["minimum_resources"] == quotes[1]["minimum_resources"]
    assert quotes[0]["expected_resource_costs"] == quotes[1]["expected_resource_costs"]
    for quote in quotes:
        assert quote["method_feedback"]["estimated_costs"] == {
            key: paid["execution_costs"][key] for key in RESOURCES}
        assert quote["minimum_resources"]["compute_cost"] >= 156042


def test_successful_actual_response_keeps_its_fresh_evidence_and_source(real_method_stream):
    stream = real_method_stream
    visit, = stream["final"]["revisits"]
    report = stream["final"]["results"][visit["task_position"]]["research"]
    outcome = report["continuation"]["method_visit_outcome"]
    pair, = report["continuation"]["method_fresh_comparisons"]
    confirmation, = outcome["response_confirmations"]
    assert pair["attempted_after_loss"] < pair["before_loss"]
    assert confirmation["source_draft_ids"] == pair["attempted_source_draft_ids"]
    assert confirmation["source_draft_ids"]
    record = stream["final"]["knowledge"]["action_experience"][confirmation["method_context"]][METHOD_RESPONSE_ACTION]
    assert record["last_after_loss"] == pair["attempted_after_loss"]
    assert record["last_gain"] > 0.
    records = {item["draft_id"]: item for item in stream["final"]["knowledge"]["research_drafts"]}
    for identifier in pair["attempted_source_draft_ids"]:
        evidence = [item for item in records[identifier]["outcomes"]
                    if item.get("kind") == "method_response_confirmation"
                    and item["probe_id"] == pair["probe_id"]]
        assert len(evidence) == 1
        assert evidence[0]["program"] == pair["attempted_program"]
        assert evidence[0]["gain"] > 0.
    assert all(outcome not in draft.get("outcomes", []) for draft in records.values())
