"""Changed research knowledge proposes and funds new tests of old paid frontiers."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.knowledge_changes import knowledge_changes, knowledge_snapshot
from hegel_machine.cognition.knowledge_revisits import offer_object_experiment
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.relation_input_diagnosis import bind_relation_entry, measured_relation_entry
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_revisits import offer_successor_knowledge
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from test_cognition_object_revisits import object_revisited_stream, recorded_factory
from test_cognition_relation_input_recovery import (
    SETTINGS, PaidInputs, command_evidence, diagnose_relation_inputs, joint_probe, setup_source,
)


CONFIG = ResearchConfig(observation_budget=96, compute_budget=12000,
    minimum_improvement=1e-7, seed=71, max_rounds=6)
TASK = {"environment": "objects", **SETTINGS, "relation_condition": "source_less", "seed": 71}
RESOURCES = ("observation_cost", "measurement_cost", "compute_cost")


def persisted(value):
    return json.loads(json.dumps(value))


@pytest.fixture(scope="module")
def updated_relation_frontier():
    purchases, projections, evaluations = [], [], []
    factory = recorded_factory(purchases, projections, evaluations)
    with factory(TASK) as lab:
        agent, origin, source_batches = setup_source(lab)
        previous = UnifiedResearchAgent(CONFIG, measurement_budget=1152,
            library=agent.library).run(lab)
        assert previous["status"] == "unresolved" and previous["selection_loss"] < 1e-7
        original_ids = {concept.concept_id for concept in agent.library.concepts}
        before = knowledge_snapshot(agent.library)
        # The later instrument experiment supplies actual new evidence for
        # the same retained relation; it does not create a response concept.
        plan, evidence = command_evidence(lab, origin, joint_probe("successor_input_evidence"))
        paid, budget = PaidInputs(lab, source_batches), SearchBudget(60000)
        diagnosis = diagnose_relation_inputs(agent, lab, origin, origin, plan, evidence,
            paid.acquire, paid.paired_for, budget, [])
        assert diagnosis["response_status"] == "supported_on_paid_rows"
        spec = lab.spec_for(("fine",))
        measured = bind_relation_entry(measured_relation_entry(diagnosis["entry"], spec, agent.library),
            {name: Program("feature", name=name) for name in spec.feature_names})
        trajectories = [[row for _, row in batch] for batch in paid.batches[len(source_batches):]]
        research = RelationIntervalResearch(measured["candidate"].program, agent.library.meta_priors,
            budget, research_program=measured["research_program"])
        alternatives = research.alternatives([row for rows in trajectories for row in rows],
            budget, experiments=trajectories)
        assert alternatives and research.evidence_contexts
        prior = agent.library.meta_priors[0]
        prior["research_program"] = alternatives[0]["research_program"]
        after = knowledge_snapshot(agent.library)
        changes = knowledge_changes(before, after)
        assert {concept.concept_id for concept in agent.library.concepts} == original_ids
        return {"previous": persisted(previous), "knowledge": persisted(agent.library.to_dict()),
                "changes": changes, "before": before, "after": after}


def test_existing_relation_evidence_changes_but_accounting_does_not(updated_relation_frontier):
    data = updated_relation_frontier
    relation, = [item for item in data["changes"] if item["kind"] == "relation"]
    assert relation["change"] == "revised" and relation["available"]
    assert "research_program" in relation["changed_fields"]
    assert relation["knowledge_id"] in data["before"]["relation"]
    library = KnowledgeLibrary.from_dict(data["knowledge"])
    before = knowledge_snapshot(library)
    for model in library.instrument_models:
        model["uses"] = model.get("uses", 0) + 1
        model["compute_cost"] = model.get("compute_cost", 0) + 10
        for field in model["fields"].values():
            field["fit_loss"] = field.get("fit_loss", 0.) + 1e-15
    assert knowledge_changes(before, knowledge_snapshot(library)) == []


def test_changed_relation_can_offer_a_real_experiment_without_old_loss_gain_or_purchase(updated_relation_frontier):
    data = updated_relation_frontier
    library = KnowledgeLibrary.from_dict(data["knowledge"])
    agent = UnifiedResearchAgent(CONFIG, library=library)
    changes = [item for item in data["changes"] if item["kind"] == "relation"]
    previous = deepcopy(data["previous"])
    purchases, projections, evaluations = [], [], []
    before = persisted(library.to_dict())
    work = SearchBudget(45000)
    with recorded_factory(purchases, projections, evaluations)(TASK) as lab:
        offer = offer_successor_knowledge(agent, lab, previous, set(), work, knowledge_changes=changes)
        assert offer is not None and offer["old_evidence_gain"] == 0.
        assert offer["basis"] == "new_knowledge_distinguishing_experiment"
        assert offer["predicted_separation"] > CONFIG.minimum_improvement
        assert offer["candidate"]["source_concept_ids"] == [changes[0]["knowledge_id"]]
        latest = [context for step in library.meta_priors[0]["research_program"]["selection_steps"]
                  for context in step.get("evidence_contexts", [])]
        assert latest
        for entry in offer["candidate"]["experiment_entries"]:
            consumed = [context for step in entry["research_program"]["selection_steps"]
                        for context in step.get("evidence_contexts", [])]
            assert all(context in consumed for context in latest)
        assert all(list(offer["candidate"]["experiment"]["probe"]["sequence"]) != batch["probe"]["sequence"]
                   for batch in previous["batches"])
        unrelated = [{**changes[0], "knowledge_id": "unrelated_concept"}]
        assert offer_object_experiment(agent, lab, previous, unrelated, SearchBudget(45000)) is None
        assert offer_object_experiment(agent, lab, previous, [], SearchBudget(45000)) is None
    assert 0 < work.spent <= work.limit
    assert purchases == evaluations == []
    assert previous == data["previous"] and persisted(library.to_dict()) == before


def test_changes_only_session_schedules_a_short_experiment_from_a_35_row_pool(updated_relation_frontier):
    data = updated_relation_frontier
    previous = deepcopy(data["previous"])
    purchases, projections, evaluations = [], [], []
    config = replace(CONFIG, observation_budget=previous["costs"]["observation_cost"] + 35,
                     compute_budget=360000)
    session = UnifiedResearchSession(config, tasks=[TASK],
        library=KnowledgeLibrary.from_dict(data["knowledge"]),
        measurement_budget=previous["costs"]["measurement_cost"] + 768,
        lab_factory=recorded_factory(purchases, projections, evaluations))
    session.state["results"].append({"task_position": 0, "research": previous})
    assert session._remaining()["observation_cost"] == 35
    changes = [item for item in data["changes"] if item["kind"] == "relation"]
    session._offer_new_knowledge(1, [], changes=changes)
    state = session.to_dict()
    offer, = state["pending_revisits"]
    assert offer["basis"] == "new_knowledge_distinguishing_experiment"
    assert offer["old_evidence_gain"] == 0.
    assert 0 < offer["minimum_resources"]["observation_cost"] <= 35
    event = state["scheduler_events"][-1]
    assert event["scheduled"] and event["source_concept_ids"] == []
    assert event["knowledge_changes"] == changes
    assert 0 < event["compute_cost"] <= 45000
    assert purchases == evaluations == []
    assert state["results"][0]["research"] == previous


def test_measured_instrument_refutation_and_a_new_method_step_are_semantic_changes():
    from hegel_machine.cognition.instrument_method_lab import InstrumentMethodLab
    from test_cognition_instrument_method_runtime import agent_for, distinguish, fit_short, methods

    agent = agent_for()
    with InstrumentMethodLab(0, 157, 739, "recurrent") as lab:
        model, batches = fit_short(agent, lab, 3)
    before = knowledge_snapshot(agent.library)
    identifier = model["model_id"]
    with InstrumentMethodLab(1, 157, 739, "recurrent") as lab:
        result, paid, _, _ = distinguish(agent, lab, model, batches)
    assert result["model_id"] == identifier and paid.purchases
    changes = knowledge_changes(before, knowledge_snapshot(agent.library))
    assert any(item["kind"] == "instrument" and item["knowledge_id"] == identifier
               and item["change"] == "revised" for item in changes)
    method_id = methods(agent)[0]["method_id"]
    assert any(item["kind"] == "method" and item["knowledge_id"] == method_id for item in changes)

    with InstrumentMethodLab(2, 881, 739, "recurrent") as lab:
        model, batches = fit_short(agent, lab, 4)
        _, paid, _, _ = distinguish(agent, lab, model, batches)
        assert not paid.purchases
    before = knowledge_snapshot(agent.library)
    with InstrumentMethodLab(3, 881, 739, "recurrent") as lab:
        result, paid, _, _ = distinguish(agent, lab, model, batches)
    assert result["status"] == "supported_on_measured_trajectory"
    assert paid.purchases[0]["probe"].cost == 5
    changes = knowledge_changes(before, knowledge_snapshot(agent.library))
    assert any(item["kind"] == "method" and item["change"] == "added"
               and item.get("parent_method_id") == method_id for item in changes)

    before = knowledge_snapshot(agent.library)
    for method in methods(agent):
        assert method["support"]
        method["uses"] = method.get("uses", 0) + 1
        method["support"].append({**deepcopy(method["support"][0]), "task_id": "another_success"})
    assert knowledge_changes(before, knowledge_snapshot(agent.library)) == []


def test_json_experiment_resume_uses_old_bank_and_charges_only_new_purchases(updated_relation_frontier):
    data = updated_relation_frontier
    previous = data["previous"]
    changes = [item for item in data["changes"] if item["kind"] == "relation"]
    library = KnowledgeLibrary.from_dict(data["knowledge"])
    purchases, projections, evaluations = [], [], []
    factory = recorded_factory(purchases, projections, evaluations)
    with factory(TASK) as lab:
        offer = offer_object_experiment(UnifiedResearchAgent(CONFIG, library=library),
            lab, previous, changes, SearchBudget(45000))
    assert offer is not None and purchases == []
    candidate = persisted(offer["candidate"])
    incumbent = Program.from_dict(previous["program"])
    assert any(Program.from_dict(item["program"]).expanded() == incumbent.expanded()
               for item in candidate["experiment"]["hypotheses"])
    config = replace(CONFIG, observation_budget=previous["costs"]["observation_cost"] + 64,
        compute_budget=previous["costs"]["compute_cost"] + 60000)
    with factory(TASK) as lab:
        current = UnifiedResearchAgent(config, measurement_budget=previous["costs"]["measurement_cost"] + 768,
            library=KnowledgeLibrary.from_dict(data["knowledge"])).run(lab,
                previous=persisted(previous), continuation_candidate=candidate)
    assert current["batches"][:len(previous["batches"])] == previous["batches"]
    events = current["actions"][len(previous["actions"]):]
    tested = next(event for event in events if event["action"] == "test_successor_object_experiment")
    acquired = next(item for item in purchases if item["probe"].probe_id == tested["probe_id"])
    assert list(acquired["probe"].sequence) == candidate["experiment"]["probe"]["sequence"]
    assert tested["evidence"]["observation_rows"] == acquired["observation_cost"] > 0
    assert tested["evidence"]["status"] == "distinguished"
    conditioned = [event for event in events if event["action"] == "condition_inherited_relation_intervals"]
    assert any(event["distinguishing_trajectories"] > 0 for event in conditioned)
    comparison = [event for event in events
                  if event["action"] == "compare_object_continuation_candidates"][-1]
    # This actual random confirmation misses the remaining radius boundary.
    # Its tie cannot give the old program the new program's research status.
    assert comparison["baseline_program"] == previous["program"]
    assert comparison["before_loss"] == pytest.approx(comparison["after_loss"], abs=1e-12)
    assert comparison["program"] != comparison["baseline_program"]
    assert current["program"] == comparison["program"]
    for resource in RESOURCES:
        assert current["costs"][resource] == previous["costs"][resource] + current["visit_costs"][resource]
        if resource != "compute_cost":
            assert current["visit_costs"][resource] == sum(item[resource] for item in purchases)
    assert 0 < current["visit_costs"]["compute_cost"] <= 60000
    assert current["visit_costs"]["observation_cost"] <= 64
    old_ids = {batch["probe"]["probe_id"] for batch in previous["batches"]}
    assert not old_ids & {item["probe"].probe_id for item in purchases}
    assert evaluations == []
    if current["status"] == "completed":
        confirm = next(index for index, event in enumerate(events)
            if event["action"] == "validate_frozen_program" and event["accepted"])
        assert events.index(tested) < confirm


def test_revisit_cannot_borrow_the_next_unvisited_tasks_budget(object_revisited_stream):
    stream = object_revisited_stream
    state = persisted(stream["queued"])
    state["tasks"].append({**state["tasks"][-1], "seed": 883,
        "research_limits": {"observation_budget": 31, "measurement_budget": 72, "compute_budget": 901}})
    session = UnifiedResearchSession(state=state, lab_factory=stream["factory"])
    before = session._remaining()
    same_visited = UnifiedResearchSession(state=stream["queued"], lab_factory=stream["factory"])
    assert before == same_visited._remaining()
    offer, = session.state["pending_revisits"]
    session.state["pending_revisits"] = []
    previous = deepcopy(session.state["results"][0]["research"])
    assert session._perform_revisit(offer)
    result = session.to_dict()
    assert len(result["results"]) == 2
    assert result["results"][1] == state["results"][1]
    for resource in RESOURCES:
        assert result["revisits"][-1]["resource_limits"][resource] <= previous["costs"][resource] + before[resource]
        assert result["results"][0]["research"]["visit_costs"][resource] <= before[resource]
