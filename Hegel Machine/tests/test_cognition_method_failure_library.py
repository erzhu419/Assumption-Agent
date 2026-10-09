"""A paid failure can request an older method without relearning its source."""

from copy import deepcopy
import gzip
import importlib
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.unified_continuation import paid_view
from hegel_machine.cognition.unified_session import UnifiedResearchSession


RESOURCES = ("observation_cost", "measurement_cost", "compute_cost")


def persisted(value):
    return json.loads(json.dumps(value))


def restore_histories(report):
    for batch in report["batches"]:
        history = []
        for row in batch["observations"]:
            row["history"] = list(history)
            history.append({**row["features"], "__target__": row["target"]})


@pytest.fixture(scope="module")
def recorded():
    path = Path(__file__).parent / "fixtures/method_failure_library_root02.json.gz"
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


@pytest.fixture(scope="module")
def laboratory_factory():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
        return importlib.import_module("run_observed_condition_decoding_study").laboratory


def paid_prefix(recorded, *, failed=False):
    state = deepcopy(recorded["before_session"])
    # End at the actual visited prefix; an unvisited successor contributes no
    # evidence, knowledge or budget to this regression.
    state["tasks"] = state["tasks"][:6]
    if failed:
        state["results"].append(deepcopy(recorded["failed_result"]))
        state["knowledge"] = deepcopy(recorded["after_failure_knowledge"])
        state["scheduler_events"].extend(deepcopy(recorded["failure_scheduler_events"]))
        state["method_revisit_attempts"].extend(deepcopy(recorded["failure_method_revisit_attempts"]))
        state["screening_compute_cost"] += recorded["failure_screening_compute_delta"]
    for result in state["results"]:
        restore_histories(result["research"])
    return state


@pytest.fixture(scope="module")
def real_streams(recorded, laboratory_factory):
    result = {}
    for name, failed in (("initial", False), ("failure", True)):
        labs = []

        def factory(task):
            lab = laboratory_factory(task)
            labs.append((deepcopy(task), lab))
            return lab

        before = paid_prefix(recorded, failed=failed)
        session = UnifiedResearchSession(state=before, lab_factory=factory)
        after = persisted(session.run(max_tasks=1))
        result[name] = {"before": before, "after": after, "labs": labs}
    print(json.dumps({"method_failure_library": {
        name: {"p5_status": stream["after"]["results"][5]["research"]["status"],
            "p5_costs": stream["after"]["results"][5]["research"]["costs"],
            "retained_events": [{"position": event["task_position"],
                "scheduled": event["scheduled"], "compute": event["compute_cost"],
                "drafts": (event.get("method_opportunity") or {}).get("draft_ids"),
                "paid_loss": (event.get("method_opportunity") or {}).get("paid_counterexample_loss"),
                "candidate_loss": (event.get("method_opportunity") or {}).get("candidate_paid_loss")}
                for event in stream["after"]["scheduler_events"]
                [len(stream["before"]["scheduler_events"]):]
                if event["action"] == "try_retained_method"],
            "revisits": [{"position": visit["task_position"],
                "costs": (visit.get("method_visit_outcome") or {}).get("whole_visit_costs"),
                "fresh_before": ((visit.get("method_visit_outcome") or {}).get("fresh_comparison") or {}).get("before_loss"),
                "fresh_after": ((visit.get("method_visit_outcome") or {}).get("fresh_comparison") or {}).get("after_loss")}
                for visit in stream["after"]["revisits"]
                [len(stream["before"]["revisits"]):]]}
        for name, stream in result.items()}}, sort_keys=True))
    return result


def test_normal_first_visit_replays_the_rich_older_response_on_current_paid_rows(real_streams):
    stream = real_streams["initial"]
    after, before = stream["after"], stream["before"]
    report = after["results"][5]["research"]
    replays = [event for event in report["actions"] if event["action"] == "replay_research_draft"]
    rich = [event for event in replays if event["draft_id"] == "research_draft_0001"]
    assert rich and any(event["response_basis_terms"] == 10 for event in rich)
    assert any(event["compute_cost"] > 0 and event["training_loss"] is not None for event in rich)
    for event in report["actions"]:
        if event["action"] == "select_research_drafts_on_paid_response":
            assert event["comparison_basis"] == "original_response"
            assert event["compute_cost"] == sum(item["compute_cost"] for item in event["alternatives"])
    assert any(event["action"] == "validate_frozen_program" for event in report["actions"])
    assert after["results"][1] == before["results"][1]
    assert not any(task["seed"] == before["tasks"][1]["seed"] and lab.costs["observation_cost"]
                   for task, lab in stream["labs"])


def test_saved_paid_failure_buys_a_priced_older_method_and_new_confirmation(
        real_streams, laboratory_factory):
    stream = real_streams["failure"]
    before, after = stream["before"], stream["after"]
    old = before["results"][5]["research"]
    report = after["results"][5]["research"]
    visits = after["revisits"][len(before["revisits"]):]
    visit, = [item for item in visits if item["task_position"] == 5]
    assert visit["basis"] == "new_partial_method_distinguishing_experiment"
    events = after["scheduler_events"][len(before["scheduler_events"]):]
    request = next(event for event in events if event["action"] == "try_retained_method"
                   and event["task_position"] == 5 and event["scheduled"])
    assert request["compute_cost"] > 0
    assert "research_draft_0001" in request["method_opportunity"]["draft_ids"]
    assert request["method_opportunity"]["candidate_paid_loss"] < request["method_opportunity"]["paid_counterexample_loss"]
    assert request["method_opportunity"]["requires_fresh_confirmation"]
    plan = visit["candidate"]["method_experiment"]
    new_actions = report["actions"][len(old["actions"]):]
    bought = next(event for event in new_actions if event["action"] == "acquire"
                  and event["probe_id"] == plan["probe"]["probe_id"])
    assert bought["purpose"] == "successor_partial_method_experiment"
    batch = next(item for item in report["batches"]
                 if item["probe"]["probe_id"] == bought["probe_id"])
    assert batch["probe"]["sequence"] == plan["probe"]["sequence"]
    assert bought["observation_cost"] == bought["measurement_cost"] == len(batch["observations"])
    outcome = visit["method_visit_outcome"]
    fresh = outcome["fresh_comparison"]
    assert fresh is not None
    acquisition = next(event for event in new_actions if event["action"] == "acquire"
                       and event["probe_id"] == fresh["probe_id"])
    assert acquisition["purpose"] == "confirm_continuation_program"
    assert fresh["probe_id"] != bought["probe_id"]
    confirmation = next(item for item in report["batches"]
                        if item["probe"]["probe_id"] == fresh["probe_id"])
    rows = [Observation(**item) for item in confirmation["observations"]]
    assert fresh["after_loss"] == pytest.approx(loss(Program.from_dict(fresh["program"]),
                                                   rows, fresh["fallback"], .5))
    with laboratory_factory(before["tasks"][5]) as lab:
        paid = [row for batch in paid_view(lab, old, old["selected_channels"]) for row in batch]
    assert loss(Program.from_dict(report["program"]), paid, report["fallback"], .5) < loss(
        Program.from_dict(old["program"]), paid, old["fallback"], .5)
    assert report["batches"][:len(old["batches"])] == old["batches"]
    for resource in RESOURCES:
        assert report["costs"][resource] == old["costs"][resource] + report["visit_costs"][resource]
        assert outcome["whole_visit_costs"][resource] == report["visit_costs"][resource] + (
            visit["screening_compute_cost"] if resource == "compute_cost" else 0)
        assert report["visit_costs"][resource] <= request["minimum_resources"][resource]
        expected = sum(item["research"]["costs"][resource] for item in after["results"])
        if resource == "compute_cost":
            expected += after["screening_compute_cost"]
        assert after["costs"][resource] == expected
    for resource in ("observation_cost", "measurement_cost"):
        assert report["visit_costs"][resource] == sum(lab.costs[resource] for task, lab in stream["labs"])
    assert report["status"] == "unresolved" or report["confirmation_loss"] <= 1e-7


def test_json_restoration_does_not_research_explained_source_or_rebuy_the_same_method(
        real_streams, laboratory_factory):
    stream = real_streams["failure"]
    before, after = stream["before"], stream["after"]
    purchases = []

    def factory(task):
        lab = laboratory_factory(task)
        purchases.append(lab)
        return lab

    resumed = persisted(UnifiedResearchSession(state=persisted(after), lab_factory=factory).run(max_tasks=1))
    assert resumed["results"][1] == before["results"][1]
    assert resumed["results"][5] == after["results"][5]
    assert resumed["screening_compute_cost"] == after["screening_compute_cost"]
    assert resumed["method_revisit_attempts"] == after["method_revisit_attempts"]
    assert resumed["revisits"] == after["revisits"]
    assert not resumed["pending_revisits"]
    assert all(lab.costs["observation_cost"] == lab.costs["measurement_cost"] == 0 for lab in purchases)
