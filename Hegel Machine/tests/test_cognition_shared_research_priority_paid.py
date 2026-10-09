"""Reprice a saved mixed queue without repeating already paid research."""

from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.numeric_method_revisits import reprice_numeric_method_offer
from hegel_machine.cognition.unified_session import UnifiedResearchSession


@pytest.fixture
def paid_queue():
    path = Path(__file__).parent / "fixtures/shared_research_priority_root09.json.gz"
    return json.loads(gzip.decompress(path.read_bytes()))


def no_new_laboratory(_arguments):
    raise AssertionError("Repricing already paid plans must not open a laboratory")


def method(offers, position):
    return next(offer for offer in offers if offer["task_position"] == position)


def expected_priority(offer, remaining):
    opportunity = offer["candidate"]["method_opportunity"]
    # This actual saved opportunity has no prior visit and zero predicted
    # separation. Its value is the measured improvement on shared paid rows.
    assert opportunity["observed_visit_trials"] == 0
    assert opportunity["predicted_separation"] == 0.
    assert opportunity["confirmation_allocation"] is None
    value = ((opportunity["paid_counterexample_loss"] - opportunity["candidate_paid_loss"])
             / opportunity["paid_counterexample_loss"])
    costs = opportunity["recommended_resources"]
    share = max(costs[key] / remaining[key] for key in costs)
    return value / share


def test_saved_useful_method_competes_with_instruments_on_the_same_pool(paid_queue):
    position = paid_queue["known_paid_repair_position"]
    old = method(paid_queue["state"]["pending_revisits"], position)
    assert max(paid_queue["state"]["pending_revisits"], key=lambda item: item["relative_gain"])["basis"] == "unresolved_instrument_condition"
    session = UnifiedResearchSession(state=paid_queue["state"], lab_factory=no_new_laboratory)
    before = deepcopy(session._costs())
    assert session._remaining() == paid_queue["remaining"]
    session._offer_response_questions()
    current = method(session.state["pending_revisits"], position)
    assert current["relative_gain"] == pytest.approx(expected_priority(old, session._remaining()))
    assert current["relative_gain"] > old["relative_gain"]
    assert max(session.state["pending_revisits"], key=lambda item: item["relative_gain"])["task_position"] == position
    assert current["candidate"] == old["candidate"]
    assert current["minimum_resources"] == old["minimum_resources"]
    assert current["old_evidence_gain"] == old["old_evidence_gain"]
    assert session._costs() == before


def test_pool_change_after_json_restore_reprices_without_refitting_or_extra_screening(paid_queue):
    position = paid_queue["known_paid_repair_position"]
    session = UnifiedResearchSession(state=paid_queue["state"], lab_factory=no_new_laboratory)
    session._offer_response_questions()
    old = deepcopy(method(session.state["pending_revisits"], position))
    # Another already visited task has incurred additional paid work. This
    # changes opportunity cost, not the candidate's research evidence.
    session.state["results"][0]["research"]["costs"]["compute_cost"] += 100000
    restored = UnifiedResearchSession(state=json.loads(json.dumps(session.to_dict())),
        lab_factory=no_new_laboratory)
    before = restored._costs()
    screening = restored.state["screening_compute_cost"]
    restored._offer_response_questions()
    current = method(restored.state["pending_revisits"], position)
    assert current["relative_gain"] == pytest.approx(expected_priority(old, restored._remaining()))
    assert current["relative_gain"] < old["relative_gain"]
    assert current["candidate"] == old["candidate"]
    assert current["minimum_resources"] == old["minimum_resources"]
    assert restored._costs() == before
    assert restored.state["screening_compute_cost"] == screening
    stable = deepcopy(current)
    restored._offer_response_questions()
    assert method(restored.state["pending_revisits"], position) == stable
    assert restored._costs() == before


@pytest.mark.parametrize("resource", ["observation_cost", "measurement_cost", "compute_cost"])
def test_repricing_never_weakens_the_paid_methods_hard_minimum(paid_queue, resource):
    position = paid_queue["known_paid_repair_position"]
    session = UnifiedResearchSession(state=paid_queue["state"], lab_factory=no_new_laboratory)
    session._offer_response_questions()
    offer = method(session.state["pending_revisits"], position)
    quoted = method(paid_queue["state"]["pending_revisits"], position)["minimum_resources"]
    assert offer["minimum_resources"] == quoted
    limits = {**quoted, resource: quoted[resource] - 1}
    before = session._costs()
    assert not session._perform_revisit(offer, allocation_limits=limits)
    assert session.state["scheduler_events"][-1]["reason"] == "remaining_resources"
    assert session._costs() == before


def test_quote_depends_on_paid_value_and_resources_not_task_labels(paid_queue):
    original = method(paid_queue["state"]["pending_revisits"], paid_queue["known_paid_repair_position"])
    one, other = deepcopy(original), deepcopy(original)
    other["task_position"], other["source_task_position"] = 71, 88
    other["candidate"]["method_opportunity"]["source_task_ids"] = ["renamed_source"]
    reprice_numeric_method_offer(one, paid_queue["remaining"])
    reprice_numeric_method_offer(other, paid_queue["remaining"])
    assert one["relative_gain"] == other["relative_gain"]
    assert one["minimum_resources"] == other["minimum_resources"] == original["minimum_resources"]
