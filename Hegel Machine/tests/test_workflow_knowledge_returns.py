"""Paid source comparisons inform workflows without becoming execution feedback."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.growth_methods import ResearchMethodBook
from hegel_machine.cognition.knowledge_returns import transfer_condition
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, Probe, SearchBudget, TaskSpec


def _setup(*, quadratic=False, producer_index=0):
    spec = TaskSpec("source", "Predict a response.", ("x",), (), (),
                    feature_bounds={"x": (-1., 1.)})
    x = Program("feature", name="x")
    program = Program("mul", children=(x, x)) if quadratic else x
    values = (-1., -.5, 0., .5, 1.)
    rows = [Observation({"x": value}, value * value if quadratic else value) for value in values]
    library = KnowledgeLibrary()
    source_id, _ = library.learn(Candidate("source", program, "fit"), spec, rows)
    diagnostics = {"residual_error": 0., "feature_count": 1,
        "feature_bounds": spec.feature_bounds, "transfer_condition": transfer_condition(spec, rows)}
    probe = Probe("source_paid", tuple(row.features for row in rows))
    trace = [{"action": "vary_controls", "probe": probe}, {"action": "fit"},
             {"action": "confirm", "probe": probe}]
    first = library.research_workflows.learn(trace, diagnostics, success=True, task_id="source")
    alternative = library.research_workflows.learn(
        [trace[0], {"action": "replicate", "probe": probe}, *trace[1:]],
        diagnostics, success=True, task_id="other_source")
    method = (first, alternative)[producer_index]
    library.concepts[0].production_methods.append({"kind": "research_workflow",
        "method_id": method["method_id"], "source_task": "source",
        "source_concept_id": source_id, "diagnostics": deepcopy(diagnostics)})
    return library, diagnostics, method["method_id"], (first, alternative)


def _consume(library, *, quadratic=False, status="completed", opposite=False, neutral=False,
             task_id="consumer", scale=1.):
    spec = TaskSpec(task_id, "Predict a response.", ("u",), (), (),
                    feature_bounds={"u": (-1., 1.)})
    response = lambda value: scale * (value * value if quadratic else value)
    rows = [Observation({"u": value}, response(value)) for value in (-1., -.5, 0., .5, 1.)]
    fresh = [Observation({"u": value}, -response(value) if opposite else response(value))
             for value in (-.8, -.3, .3, .8)]
    library.begin_transfer_episode(spec, rows)
    budget = SearchBudget(1000)
    candidates = library.propose(spec, rows, budget, preserve_parameters=False)
    winner = min(candidates, key=lambda item: loss(item.program, fresh, 0., .5))
    baseline = loss(winner.program, fresh, 0., .5) if neutral else loss(
        Program("const", value=0.), fresh, 0., .5)
    for candidate in candidates:
        measured = loss(candidate.program, fresh, 0., .5)
        library.record_transfer_result(candidate, spec,
            accepted=not opposite and candidate == winner, loss=measured,
            baseline_loss=baseline, evidence_rows=len(fresh),
            comparison_compute=len(fresh), comparison_id="fresh")
    final = (Candidate("independent", Program("mul", children=(
        Program("const", value=-1.), Program("feature", name="u"))), "refit")
        if opposite else winner)
    result = library.finish_transfer_episode(spec, final_candidate=final,
        final_loss=loss(final.program, fresh, 0., .5), baseline_loss=baseline,
        compute_cost=budget.spent + len(candidates) * len(fresh),
        observation_cost=len(rows) + len(fresh), validation_rows=len(fresh), final_status=status)
    return result


def test_confirmed_paired_output_gain_changes_only_matching_workflow_choice_and_survives_json():
    library, diagnostics, producer, methods = _setup()
    before = deepcopy(library.research_workflows.methods)
    assert library.research_workflows.propose(diagnostics)["method_id"] == methods[1]["method_id"]
    result = _consume(library)
    outcome = result["method_returns"][0]
    assert outcome["retained_gain"] == pytest.approx(1.)
    assert outcome["source_returns"][0]["confirmation_status"] == "confirmed"
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    assert restored.research_workflows.propose(diagnostics)["method_id"] == producer
    for old, current in zip(before, restored.research_workflows.methods):
        assert current["examples"] == old["examples"]
        assert current["applicability"] == old["applicability"]
    changed_spec = TaskSpec("changed", "Predict a response.", ("x",), (), (),
                            feature_bounds={"x": (-1., 1.)})
    changed_rows = [Observation({"x": value}, 2 * value) for value in (-1., -.5, 0., .5, 1.)]
    changed = {**diagnostics, "transfer_condition": transfer_condition(changed_spec, changed_rows)}
    assert restored.research_workflows.propose(changed)["method_id"] == methods[1]["method_id"]


@pytest.mark.parametrize("status,neutral", [("unresolved", False), ("completed", True)])
def test_provisional_or_neutral_gain_does_not_promote_a_workflow(status, neutral):
    library, diagnostics, _, methods = _setup()
    result = _consume(library, status=status, neutral=neutral)
    outcome = result["method_returns"][0]
    assert outcome["retained_gain"] is None
    assert outcome["source_returns"][0]["confirmation_status"] == (
        "neutral" if neutral else "provisional")
    assert library.research_workflows.propose(diagnostics)["method_id"] == methods[1]["method_id"]


def test_later_source_confirmation_refreshes_earlier_provisional_workflow_receipts():
    library, diagnostics, producer, methods = _setup(quadratic=True)
    first = _consume(library, quadratic=True)
    assert first["method_returns"][0]["retained_gain"] is None
    assert library.research_workflows.propose(diagnostics)["method_id"] == methods[1]["method_id"]
    _consume(library, quadratic=True)
    outcomes = library.research_workflows.methods[0]["knowledge_outcomes"]
    assert len(outcomes) == 2
    assert all(item["retained_gain"] > 0 for item in outcomes)
    assert all(receipt["confirmation_status"] == "confirmed"
               for item in outcomes for receipt in item["source_returns"])
    assert library.research_workflows.propose(diagnostics)["method_id"] == producer


def test_negative_output_return_does_not_globally_reject_its_producing_method():
    library, diagnostics, producer, methods = _setup(producer_index=1)
    result = _consume(library, opposite=True)
    assert result["method_returns"][0]["retained_gain"] < 0
    assert library.research_workflows.propose(diagnostics)["method_id"] == methods[0]["method_id"]
    spec = TaskSpec("new_region", "Predict a response.", ("x",), (), (),
                    feature_bounds={"x": (-2., 2.)})
    rows = [Observation({"x": value}, value) for value in (-2., -1., 0., 1., 2.)]
    changed = {**diagnostics, "feature_bounds": spec.feature_bounds,
               "transfer_condition": transfer_condition(spec, rows)}
    assert library.research_workflows.propose(changed)["method_id"] == producer
    assert library.research_workflows.methods[1]["status"] == "supported"
    assert len(library.research_workflows.methods[1]["examples"]) == 1


def test_shared_joint_offer_keeps_measurement_but_gives_no_exclusive_workflow_credit():
    library, diagnostics, producer, methods = _setup()
    x = Program("feature", name="x")
    spec = TaskSpec("second_source", "Predict a response.", ("x",), (), (),
                    feature_bounds={"x": (-1., 1.)})
    second, _ = library.learn(Candidate("second", Program("mul", children=(
        Program("const", value=2.), x)), "fit"), spec,
        [Observation({"x": value}, 2 * value) for value in (-1., 0., 1.)])
    consumer = replace(spec, task_id="consumer", feature_names=("u",), feature_bounds={"u": (-1., 1.)})
    rows = [Observation({"u": value}, 3 * value) for value in (-1., -.5, 0., .5, 1.)]
    fresh = [Observation({"u": value}, 3 * value) for value in (-.8, -.3, .3, .8)]
    library.begin_transfer_episode(consumer, rows)
    budget = SearchBudget(1000)
    offers = library.propose(consumer, rows, budget, preserve_parameters=True)
    first_program = x.remap_features({"x": "u"})
    second_program = Program("mul", children=(Program("const", value=2.), first_program))
    selected = [next(item for item in offers if item.memory_id == source
                     and item.program.expanded().simplified() == program)
                for source, program in ((library.concepts[0].concept_id, first_program), (second, second_program))]
    combined = Program("add", children=tuple(item.program for item in selected))
    source_ids = [item.memory_id for item in selected]
    measured = loss(combined, fresh, 0., .5)
    baseline = loss(Program("const", value=0.), fresh, 0., .5)
    for index, offer in enumerate(selected):
        candidate = Candidate(f"joint_{index}", combined, "compose", parents=tuple(source_ids),
                              memory_id=offer.memory_id)
        origin = {**library.transfer_evidence(offer)[0], "lineage_id": "paid_joint",
                  "joint_source_ids": source_ids}
        library.record_transfer_result(candidate, consumer, accepted=True, loss=measured,
            baseline_loss=baseline, evidence_rows=len(fresh), comparison_compute=len(fresh),
            comparison_id="joint_fresh", origin=origin)
    result = library.finish_transfer_episode(consumer, final_candidate=candidate,
        final_loss=measured, baseline_loss=baseline, compute_cost=budget.spent + len(fresh),
        observation_cost=len(rows) + len(fresh), validation_rows=len(fresh), final_status="completed")
    outcome = result["method_returns"][0]
    assert outcome["shared_gain"] > 0
    assert outcome["retained_gain"] is None
    assert all(not item["exclusive"] for item in outcome["comparisons"])
    assert library.research_workflows.propose(diagnostics)["method_id"] == methods[1]["method_id"]


def test_deleted_workflow_identity_is_not_reused_for_an_existing_concepts_producer():
    library, diagnostics, removed, _ = _setup()
    state = library.to_dict()
    # Removing learned procedures retains the identity counter and concepts.
    state["research_workflows"]["methods"] = []
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(state)))
    fresh = restored.research_workflows.learn([{"action": "fit"}, {"action": "confirm"}],
        diagnostics, success=True, task_id="new_producer")
    assert fresh["method_id"] != removed
    result = _consume(restored)
    assert result["returns"] and result["method_returns"] == []
    assert "knowledge_outcomes" not in restored.research_workflows.methods[0]


def test_old_method_book_without_counter_preserves_existing_identity_after_revision():
    library, diagnostics, _, methods = _setup()
    old = {"methods": library.research_workflows.to_dict()["methods"]}
    restored = ResearchMethodBook.from_dict(json.loads(json.dumps(old)))
    child = restored.revise(methods[1]["method_id"],
        [{"action": "fit"}, {"action": "confirm"}], diagnostics, success=True, task_id="revision")
    assert child["method_id"] not in {item["method_id"] for item in methods}
    assert child["parent_method_id"] == methods[1]["method_id"]


def test_new_paid_visit_with_same_public_task_label_returns_to_existing_workflow():
    library, diagnostics, producer, _ = _setup()
    result = _consume(library, task_id="source")
    outcome = result["method_returns"][0]
    assert outcome["production_task"] == outcome["downstream_task"] == "source"
    assert outcome["retained_gain"] > 0
    assert library.research_workflows.propose(diagnostics)["method_id"] == producer


def test_distinct_paid_consumer_demand_can_select_its_producing_workflow():
    library, diagnostics, producer, methods = _setup()
    result = _consume(library, scale=2.)
    condition = result["returns"][0]["condition"]
    assert condition != diagnostics["transfer_condition"]
    assert library.research_workflows.propose(diagnostics)["method_id"] == methods[1]["method_id"]
    demand = {"task_position": 7, "condition": condition, "paid_rows": 5, "status": "unresolved"}
    context = {**diagnostics, "downstream_demands": [demand]}
    chosen = library.research_workflows.propose(context)
    assert chosen["method_id"] == producer
    value = library.research_workflows.knowledge_value(chosen, context)
    assert value["priority"][0] < 0
    assert value["selected_demand"]["task_position"] == 7
    assert value["matched_demands"][0]["receipts"][0]["confirmation_status"] == "confirmed"
    multiple = {**context, "downstream_demands": [demand, {**demand, "task_position": 8}]}
    assert library.research_workflows.knowledge_value(chosen, multiple)["priority"] == value["priority"]
    # The executor can exclude an unfunded plan before economic ranking.
    allowed = [library.research_workflows.methods[1]]
    assert library.research_workflows.propose(context, candidates=allowed)["method_id"] == methods[1]["method_id"]


@pytest.mark.parametrize("demand_kind", ["empty", "solved", "unpaid", "unmatched"])
def test_absent_active_matching_demand_does_not_supply_economic_credit(demand_kind):
    library, diagnostics, producer, methods = _setup()
    result = _consume(library, scale=2.)
    demand = {"task_position": 7, "condition": result["returns"][0]["condition"],
              "paid_rows": 5, "status": "unresolved"}
    if demand_kind == "solved":
        demand["status"] = "completed"
    elif demand_kind == "unpaid":
        demand["paid_rows"] = 0
    elif demand_kind == "unmatched":
        demand["condition"] = diagnostics["transfer_condition"]
    context = {**diagnostics, "downstream_demands": [] if demand_kind == "empty" else [demand]}
    value = library.research_workflows.knowledge_value(library.research_workflows.methods[0], context)
    assert value["priority"] == (0., 0.)
    assert value["selected_demand"] is None
    assert library.research_workflows.propose(context)["method_id"] == methods[1]["method_id"]


def test_unknown_demand_does_not_erase_negative_value_for_a_measured_demand():
    library, diagnostics, producer, methods = _setup(producer_index=1)
    result = _consume(library, opposite=True)
    unknown_spec = TaskSpec("unknown_demand", "Predict a response.", ("u",), (), (),
                            feature_bounds={"u": (-1., 1.)})
    unknown_rows = [Observation({"u": value}, 3 * value) for value in (-1., -.5, 0., .5, 1.)]
    context = {**diagnostics, "downstream_demands": [
        {"task_position": 7, "condition": result["returns"][0]["condition"], "paid_rows": 5},
        {"task_position": 8, "condition": transfer_condition(unknown_spec, unknown_rows), "paid_rows": 5}]}
    value = library.research_workflows.knowledge_value(library.research_workflows.methods[1], context)
    assert value["priority"][0] > 0
    assert value["selected_demand"]["task_position"] == 7
    assert len(value["ignored_demands"]) == 1
    assert library.research_workflows.propose(context)["method_id"] == methods[0]["method_id"]
    assert library.research_workflows.methods[1]["status"] == "supported"


def test_explicit_demand_exposes_provisional_receipt_without_valuing_it():
    library, diagnostics, producer, methods = _setup()
    result = _consume(library, scale=2., status="unresolved")
    context = {**diagnostics, "downstream_demands": [{"task_position": 7,
        "condition": result["returns"][0]["condition"], "paid_rows": 5}]}
    value = library.research_workflows.knowledge_value(library.research_workflows.methods[0], context)
    assert value["priority"] == (0., 0.) and value["selected_demand"] is None
    assert not value["matched_demands"][0]["actionable"]
    assert value["matched_demands"][0]["receipts"][0]["confirmation_status"] == "provisional"
    assert library.research_workflows.propose(context)["method_id"] == methods[1]["method_id"]
