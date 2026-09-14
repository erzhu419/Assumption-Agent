"""Production persistence and method replay retain executable parameter evidence."""

from copy import deepcopy
from dataclasses import replace
import json

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import SearchBudget, TaskSpec
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from hegel_machine.cognition.research_programs import replay_research_program
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_cognition_retained_relation_evidence import consumer_call, physical_rows, saved_source
from test_cognition_relation_trajectory_evidence import measured_law, mse


def test_equivalent_concept_retains_new_paid_exclusions_through_actual_library_json():
    program = measured_law()
    original = deepcopy(program.to_dict())
    budget = SearchBudget(80000)
    initial = physical_rows([1.2, .5, .5, .5, .5, 1.2, 1.2, 1.2])
    research = RelationIntervalResearch(program, [], budget,
        structure={"fits": [{"radius": .8}], "radius_interval": [.2, 1.]})
    first = research.alternatives(initial, budget)[0]
    research.adopt(first)
    spec = TaskSpec("same_response_new_evidence", "", tuple(sorted(program.required_features())), (), ())
    agent = UnifiedResearchAgent(library=KnowledgeLibrary())
    concept, added = agent.library.learn(first["candidate"], spec, initial)
    assert added
    prior = {"source_concept_id": concept, "status": "supported",
             "research_program": deepcopy(first["research_program"])}
    agent.library.meta_priors.append(prior)
    old_record = deepcopy(prior)

    acquired = physical_rows([1.2, .625, .625, .625, .625, .875, .875, .875, .875, 1.2])
    latest = research.alternatives([*initial, *acquired], budget, experiments=[acquired])[0]
    same, added = agent.library.learn(latest["candidate"], spec, [*initial, *acquired])
    assert same == concept and not added
    assert agent._remember_relation_research(prior, latest, spec, [])
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(agent.library.to_dict())))
    returned = consumer_call(latest, prefix="next_", outer=False)
    returned = replace(returned, definition=replace(returned.definition, source_concept_id=concept))
    pilot = physical_rows([1.2, .1, .1, .1, 1.2, 1.2] * 2, prefix="next_")
    new_probe = physical_rows([1.2, .625, .625, .625, .625, .875, .875, .875, .875, 1.2],
                             prefix="next_")
    before = RelationIntervalResearch(returned, [old_record], budget).alternatives(pilot, budget)
    after = RelationIntervalResearch(returned, restored.meta_priors, budget).alternatives(pilot, budget)
    assert before and after
    assert any(mse(item["candidate"].program, new_probe) > 1e-7 for item in before)
    assert all(mse(item["candidate"].program, new_probe) < 1e-16 for item in after)
    assert all(mse(item["candidate"].program, pilot) < 1e-16 for item in after)
    assert program.to_dict() == original and budget.spent <= budget.limit


def test_refitting_a_saved_method_preserves_exclusions_and_can_reopen_them_on_new_evidence():
    _, _, _, record = saved_source()
    untouched = deepcopy(record)
    features = {name: Program("feature", name="next_" + name)
                for name in ("f0", "x0", "q0", "f1", "x1", "q1")}
    pilot = physical_rows([1.2, .1, .1, .1, 1.2, 1.2, 1.2] * 2, prefix="next_")
    budget = SearchBudget(70000)
    replayed = replay_research_program(record["research_program"], pilot, budget,
                                       feature_mapping=features)
    assert replayed is not None and mse(replayed["candidate"].program, pilot) < 1e-7
    inherited = RelationIntervalResearch(replayed["candidate"].program, [], budget,
                                         research_program=replayed["research_program"])
    same_mechanism = inherited.alternatives(pilot, budget)
    assert same_mechanism
    # Current pilot data leave the radius open. The saved source trajectory
    # still rules out .7 and below after actual coefficient refitting/rename.
    assert all(next(iter(item["parameter_values"].values())) > .7 for item in same_mechanism)
    assert all(mse(item["candidate"].program, pilot) < 1e-7 for item in same_mechanism)

    changed = physical_rows([1.2, .6, .6, .6, .6, 1.2, .45, .45, .45, .45, 1.2, 1.2],
                            radius=.5, prefix="next_")
    assert all(mse(item["candidate"].program, changed) > 1e-7 for item in same_mechanism)
    revised = inherited.alternatives(changed, budget, experiments=[changed])
    assert revised and inherited.evidence_updates
    assert all(.45 < next(iter(item["parameter_values"].values())) < .6 for item in revised)
    assert all(mse(item["candidate"].program, changed) < 1e-7 for item in revised)
    assert record == untouched and budget.spent <= budget.limit
