"""Current evidence can reopen a saved relation inside its complete consumer."""

from copy import deepcopy
from dataclasses import replace
from functools import reduce
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from test_cognition_retained_relation_evidence import consumer_call, physical_rows, saved_source
from test_cognition_relation_trajectory_evidence import mse


@pytest.mark.parametrize("coordinate_scale", [1., 2.])
def test_changed_relation_reopens_beyond_source_domain_inside_renamed_consumer(coordinate_scale):
    source, source_rows, selected, prior = saved_source()
    original = deepcopy((source.to_dict(), selected["research_program"], prior))
    # Save/reload before consuming the old event/window law in an additional
    # window and total response. The second case binds changed measured x.
    consumer = consumer_call(selected, coordinate_scale=coordinate_scale)
    consumer = Program.from_dict(json.loads(json.dumps(consumer.to_dict())))
    distances = [1.8, 1.3, 1.3, 1.3, 1.3, 1.8, 1.5, 1.5, 1.5, 1.5, 1.8, 1.8]
    paid = physical_rows([value / coordinate_scale for value in distances], radius=1.4,
        coordinate_scale=coordinate_scale, prefix="next_", consumer=True)
    assert mse(consumer, paid) > 1e-7
    budget = SearchBudget(70000)
    research = RelationIntervalResearch(consumer, [prior], budget)
    assert max(high for parameter in research.parameters
               for _, high in parameter["prior_radius_intervals"]) == 1.
    answers = research.alternatives(paid, budget, tolerance=1e-12, experiments=[paid])

    assert answers and research.evidence_updates
    assert all(1.3 < next(iter(item["parameter_values"].values())) < 1.5 for item in answers)
    fresh = physical_rows([value / coordinate_scale for value in
        [1.9, 1.2, 1.2, 1.2, 1.2, 1.7, 1.7, 1.7, 1.7]], radius=1.4,
        coordinate_scale=coordinate_scale, prefix="next_", consumer=True)
    assert all(mse(item["candidate"].program, paid) < 1e-20 for item in answers)
    assert all(mse(item["candidate"].program, fresh) < 1e-20 for item in answers)
    assert all(item["candidate"].program.history_depth == consumer.history_depth for item in answers)
    assert all(item["candidate"].program.required_features() == consumer.required_features()
               for item in answers)
    assert (source.to_dict(), selected["research_program"], prior) == original
    assert mse(selected["candidate"].program, source_rows) < 1e-20
    assert 0 < budget.spent <= budget.limit


def test_supported_complete_consumer_keeps_exact_program_and_historical_domain():
    source, source_rows, selected, prior = saved_source()
    consumer = consumer_call(selected)
    original = deepcopy((source.to_dict(), prior))
    paid = physical_rows([1.2, .7, .7, .7, .7, 1.2, .9, .9, .9, .9, 1.2, 1.2],
                         prefix="next_", consumer=True)
    budget = SearchBudget(50000)
    research = RelationIntervalResearch(consumer, [prior], budget)
    answers = research.alternatives(paid, budget, tolerance=1e-12, experiments=[paid])

    assert answers and answers[0]["candidate"].program == consumer
    assert answers[0]["parameter_values"] == research.original
    assert not research.evidence_updates
    assert all(high <= 1. for parameter in research.parameters
               for _, high in parameter["radius_intervals"])
    assert (source.to_dict(), prior) == original
    assert mse(selected["candidate"].program, source_rows) < 1e-20


def test_copies_of_same_saved_relation_are_repaired_together_in_complete_response():
    source, source_rows, selected, prior = saved_source()
    original = deepcopy((source.to_dict(), prior))
    consumer = consumer_call(selected)
    # Program recomposition can select one saved callable in several terms.
    # All four copies here refer to the same physical relation and input map.
    terms = [Program("mul", children=(Program("const", value=weight), consumer))
             for weight in (.1, .2, .3, .4)]
    combined = reduce(lambda left, right: Program("add", children=(left, right)), terms)
    paid = physical_rows([1.8, 1.3, 1.3, 1.3, 1.3, 1.8, 1.5, 1.5, 1.5, 1.5, 1.8, 1.8],
                         radius=1.4, prefix="next_", consumer=True)
    budget = SearchBudget(70000)
    research = RelationIntervalResearch(combined, [prior], budget)
    assert mse(combined, paid) > 1e-7
    answers = research.alternatives(paid, budget, tolerance=1e-12, experiments=[paid])

    assert answers
    assert all(mse(item["candidate"].program, paid) < 1e-20 for item in answers)
    fresh = physical_rows([1.9, 1.2, 1.2, 1.2, 1.2, 1.7, 1.7, 1.7, 1.7],
                         radius=1.4, prefix="next_", consumer=True)
    assert all(mse(item["candidate"].program, fresh) < 1e-20 for item in answers)
    assert (source.to_dict(), prior) == original
    assert mse(selected["candidate"].program, source_rows) < 1e-20
    assert 0 < budget.spent <= budget.limit


def test_same_source_with_distinct_coordinate_bindings_keeps_independent_parameters():
    _, _, selected, prior = saved_source()
    first = consumer_call(selected, outer=False)
    second = consumer_call(selected, coordinate_scale=2., outer=False)
    consumer = Program("add", children=(first, second))
    distances = [1.8, .3, .3, .3, .3, 1.8, .6, .6, .6, .6, 1.8,
                 1.3, 1.3, 1.3, 1.3, 1.8, 1.5, 1.5, 1.5, 1.5, 1.8, 1.8]
    changed = physical_rows(distances, radius=1.4, prefix="next_")
    unchanged = physical_rows(distances, coordinate_scale=2., prefix="next_")
    paid = [replace(left, target=left.target + right.target)
            for left, right in zip(changed, unchanged)]
    budget = SearchBudget(70000)
    research = RelationIntervalResearch(consumer, [prior], budget)
    assert len(research.parameters) == 2
    first_symbol, second_symbol = [item["symbol"] for item in research.parameters]
    answers = research.alternatives(paid, budget, tolerance=1e-12, experiments=[paid])

    assert answers
    assert all(1.3 < item["parameter_values"][first_symbol] < 1.5 for item in answers)
    assert all(item["parameter_values"][second_symbol] == research.original[second_symbol]
               for item in answers)
    assert all(mse(item["candidate"].program, paid) < 1e-20 for item in answers)
    assert 0 < budget.spent <= budget.limit


def test_expanding_relation_domain_cannot_hide_changed_nonrelation_response():
    source, _, selected, prior = saved_source()
    consumer = consumer_call(selected)
    original = deepcopy((source.to_dict(), prior))
    paid = physical_rows([1.8, 1.3, 1.3, 1.3, 1.3, 1.8, 1.5, 1.5, 1.5, 1.5, 1.8, 1.8],
                         radius=1.4, prefix="next_", consumer=True)
    # A changed quantity-total coefficient changes even zero-event readouts.
    # Changing radius alone cannot explain the complete new response.
    paid = [replace(row, target=row.target + .14) for row in paid]
    budget = SearchBudget(70000)
    research = RelationIntervalResearch(consumer, [prior], budget)
    answers = research.alternatives(paid, budget, tolerance=1e-12, experiments=[paid])

    assert not answers and research.status == "refuted"
    assert (source.to_dict(), prior) == original
    assert 0 < budget.spent < budget.limit


def test_reopened_domain_uses_measured_history_when_current_frames_hide_the_edge():
    _, _, selected, prior = saved_source()
    consumer = consumer_call(selected)
    paid = [physical_rows([1.8, distance, distance, distance, distance, 1.8],
        radius=1.4, prefix="next_", consumer=True)[-1] for distance in (1.3, 1.5)]
    assert all(row.features["next_x1"] == 1.8 for row in paid)
    assert paid[0].target != paid[1].target
    budget = SearchBudget(70000)
    research = RelationIntervalResearch(consumer, [prior], budget)
    answers = research.alternatives(paid, budget, tolerance=1e-12, experiments=[paid])

    assert answers
    assert all(item["radius_interval"] == [1.3, 1.5] for item in research.intervals)
    assert all(mse(item["candidate"].program, paid) < 1e-20 for item in answers)
    assert 0 < budget.spent <= budget.limit


def test_learned_temporal_call_seeds_joint_search_after_instrument_rename():
    from hegel_machine.cognition.research_programs import _entry
    from hegel_machine.cognition.response_basis import relation_free_response_terms
    from hegel_machine.cognition.structure_lab import StructureLab
    from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
    from hegel_machine.cognition.unified_structure import propose_structures

    source = StructureLab(readout="window_total_event", seed=71)
    acquired = [row for seed in (89, 90, 91)
                for row in source.acquire(source.random_probe(seed, length=24)).observations]
    selected = propose_structures(source.spec, acquired, SearchBudget(200000))[0]
    original = deepcopy(selected["candidate"].program.to_dict())
    inherited = consumer_call(selected, prefix="measured_new_", outer=False)
    inherited = Program.from_dict(json.loads(json.dumps(inherited.to_dict())))
    target = UnifiedObjectLab(readout="window_total_event", seed=99, rename_prefix="new_")
    training = [row for seed in (317, 318, 319)
                for row in target.acquire(target.random_probe(seed, length=24), ("fine",)).observations]
    current_loss = mse(inherited, training)
    assert current_loss > 1e-7
    parent = _entry({"program": inherited, "terms": [inherited], "fit_terms": [inherited],
        "coefficients": [0., 1.], "loss": current_loss}, [{"operation": "fit_saved_call",
        "question": "Retain the learned response", "program": inherited, "loss": current_loss}],
        len(training), 0)
    budget = SearchBudget(80000)
    terms = relation_free_response_terms(inherited, budget)

    assert terms and any(term.op == "window_sum" for term in terms)
    assert all(name.startswith("measured_new_") for term in terms for name in term.required_features())
    assert all("neighbor_sum" not in term.describe() for term in terms)
    proposals = propose_structures(target.spec_for(("fine",)), training, budget, parent=parent)
    revised = min(proposals, key=lambda entry: entry["training_loss"])
    fresh = [row for seed in (1901, 1902)
             for row in target.acquire(target.random_probe(seed, length=48), ("fine",)).observations]
    assert mse(revised["candidate"].program, fresh) < 1e-7
    assert revised["structure"]["fits"][0]["family"] == "joint_window_event"
    assert revised["structure"]["radius_interval"][1] < selected["structure"]["radius_interval"][0]
    assert selected["candidate"].program.to_dict() == original
    assert 0 < budget.spent <= budget.limit


@pytest.fixture(scope="module")
def learned_source_library():
    from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent

    config = ResearchConfig(observation_budget=384, compute_budget=240000, max_rounds=6,
        minimum_improvement=1e-7, seed=99434001)
    agent = UnifiedResearchAgent(config, measurement_budget=4608)
    with AnonymousObjectLab(objects=4, readout="window_total_event", profile="confirmation",
            apparatus_seed=99431001, name_seed=99432001, wiring_seed=99433001,
            actuator_curvature=.12, relation_radius=.9687842617019095, seed=99434001) as lab:
        source = agent.run(lab)
    assert source["status"] == "completed"
    return json.loads(json.dumps(agent.library.to_dict()))


def test_random_confirmation_does_not_discard_unresolved_relation_rivals(learned_source_library):
    from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent

    config = ResearchConfig(observation_budget=256, compute_budget=180000, max_rounds=6,
        minimum_improvement=1e-7, seed=99534001)
    library = KnowledgeLibrary.from_dict(deepcopy(learned_source_library))
    with AnonymousObjectLab(objects=4, readout="window_total_event", profile="confirmation",
            apparatus_seed=99531001, name_seed=99532001, wiring_seed=99533001,
            actuator_curvature=.31, relation_radius=.74, seed=99534001) as lab:
        report = UnifiedResearchAgent(config, library=library, measurement_budget=3072).run(lab)

    actions = report["actions"]
    assert report["status"] == "unresolved"
    assert any(action["action"] == "plan_inherited_relation_experiment" for action in actions)
    retained = [action for action in actions if action["action"] == "retain_relation_competition"]
    assert retained and retained[-1]["reason"] == "relation_experiment_left_candidates_unresolved"
    assert retained[-1]["surviving_candidates"]
    confirmations = [action for action in actions if action["action"] == "validate_frozen_program"]
    assert confirmations and not confirmations[-1]["accepted"]
    assert confirmations[-1]["loss"] < 1e-7
    assert report["costs"]["observation_cost"] <= 256
    assert report["costs"]["compute_cost"] <= 180000
    assert report["costs"]["measurement_cost"] <= 3072
