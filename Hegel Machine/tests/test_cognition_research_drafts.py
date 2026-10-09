"""Unfinished research must cross a process boundary without becoming a fact."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from hegel_machine.cognition.research_drafts import remember_research_draft, replay_research_drafts
from hegel_machine.cognition.structure_lab import ObjectSchema
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_cognition_composed_relation_reopening import learned_source_library
from test_cognition_retained_relation_evidence import physical_rows
from test_cognition_relation_trajectory_evidence import measured_law, mse


def specification(prefix="", task_id="source"):
    objects = ("left", "right")
    schema = ObjectSchema(objects,
        {obj: {role: f"{prefix}{role}{index}" for role in ("f", "x", "q")}
         for index, obj in enumerate(objects)},
        {obj: (obj,) for obj in objects}, {obj: {} for obj in objects})
    return TaskSpec(task_id, "Measured exchange response", schema.feature_names, (), (),
                    context={"object_schema": json.dumps(schema.to_dict())})


def unfinished_procedure():
    library = KnowledgeLibrary()
    spec = specification()
    paid = physical_rows([1.2, .7, .7, .7, .7, 1.2, 1.2, 1.2, 1.2])
    source = measured_law()
    source_id, _ = library.learn(Candidate("measured_source", source.expanded(), "exchange"), spec, paid)
    source = replace(source, definition=replace(source.definition, source_concept_id=source_id))
    budget = SearchBudget(30000)
    research = RelationIntervalResearch(source, [], budget,
        structure={"fits": [{"radius": .8}], "radius_interval": [.2, 1.]})
    alternatives = research.alternatives(paid, budget, tolerance=1e-12, experiments=[paid])
    assert len(alternatives) > 1
    selected = alternatives[len(alternatives) // 2]
    record = remember_research_draft(library, selected, spec, [])
    assert record is not None
    return library, source_id, paid, selected, json.loads(json.dumps(record))


def test_saved_unfinished_procedure_refits_current_response_without_old_target_leakage():
    library, source_id, source_rows, selected, record = unfinished_procedure()
    original = json.loads(json.dumps(library.to_dict()))
    assert len(library.concepts) == 1 and record["status"] == "unresolved"
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(original)))
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-10), library=library)
    spec = specification("next_", "new_readout")
    training = [replace(row, target=2. * row.target + .4)
                for row in physical_rows([1.2, .7, .7, .7, .7, 1.2, .9, .9, .9, .9, 1.2],
                                         prefix="next_")]
    budget, actions = SearchBudget(10000), []
    claims = replay_research_drafts(agent, spec, training, budget, actions)
    assert claims and all(claim["research_draft_id"] == record["draft_id"] for claim in claims)
    selected_claim = min(claims, key=lambda claim: claim["training_loss"])
    future = [replace(row, target=2. * row.target + .4)
              for row in physical_rows([1.2, .7, .7, .7, 1.2, 1.2, 1.2], prefix="next_")]
    assert mse(selected_claim["candidate"].program, future) < 1e-15
    assert all(name.startswith("next_") for name in selected_claim["candidate"].program.required_features())
    step = next(item for item in selected_claim["research_program"]["selection_steps"]
                if item["operation"] == "resolve_relation_intervals")
    original_step = next(item for item in record["research_program"]["selection_steps"]
                         if item["operation"] == "resolve_relation_intervals")
    assert step["evidence_contexts"] == original_step["evidence_contexts"]
    assert max(abs(row.target) for row in source_rows) < .01
    assert min(row.target for row in training) > .39
    assert json.loads(json.dumps(library.to_dict())) == original
    assert mse(selected["candidate"].program, source_rows) < 1e-20
    assert 0 < budget.spent <= budget.limit
    assert actions and all(action["status"] == "requires_current_validation" for action in actions)


def test_current_counterexample_reopens_draft_interval_without_revoking_source():
    library, source_id, old_rows, selected, record = unfinished_procedure()
    source_program = deepcopy(library.concepts[0].to_dict())
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-12), library=restored)
    changed = physical_rows([1.2, .6, .6, .6, .6, 1.2, .45, .45, .45, .45, 1.2, 1.2],
                            radius=.5, prefix="changed_")
    # Refit on sparse current evidence before the discriminating experiment.
    pilot = physical_rows([1.2, .1, .1, .1, .1, 1.2, 1.2, 1.2], radius=.5, prefix="changed_")
    claims = replay_research_drafts(agent, specification("changed_", "changed_relation"),
                                   pilot, SearchBudget(10000), [])
    assert claims
    parent = min(claims, key=lambda item: item["training_loss"])
    assert mse(parent["candidate"].program, changed) > 1e-7
    budget = SearchBudget(70000)
    research = RelationIntervalResearch(parent["candidate"].program, [], budget,
                                        research_program=parent["research_program"])
    revised = research.alternatives(changed, budget, tolerance=1e-12, experiments=[changed])
    assert revised and research.evidence_updates
    assert all(mse(item["candidate"].program, changed) < 1e-15 for item in revised)
    assert all(.45 < next(iter(item["parameter_values"].values())) < .6 for item in revised)
    assert restored.concepts[0].to_dict() == source_program
    assert restored.research_drafts[0] == record
    assert mse(selected["candidate"].program, old_rows) < 1e-20
    # Parameter replacement opens a call into measured computations. That does
    # not remove the scientific dependency on the parent research procedure.
    child = {**revised[0], "research_draft_id": parent["research_draft_id"]}
    descendant = remember_research_draft(restored, child,
                                        specification("changed_", "changed_relation"), [])
    assert descendant["parent_draft_id"] == record["draft_id"]
    assert source_id in descendant["source_ids"] and None not in descendant["source_ids"]
    assert len(restored.research_drafts) == 2
    assert not restored.without({source_id}).research_drafts


def test_removing_source_removes_dependent_draft_but_keeps_independent_calibration():
    from test_cognition_instrument_field_repair import learned, specifications

    library, source_id, _, _, record = unfinished_procedure()
    assert source_id in record["source_ids"]
    # The calibration has its own observations and no dependence on the source
    # response concept being removed.
    instrument_learner = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7, max_history=1),
                                              library=library)
    instrument_learner._learn_instruments = True
    command, measured = specifications()
    learned(instrument_learner, command, measured)
    assert library.instrument_models
    without = library.without({source_id})
    assert not without.research_drafts and not without.concepts
    assert without.instrument_models == library.instrument_models
    assert len(library.research_drafts) == 1 and len(library.concepts) == 1


def test_real_unconfirmed_run_preserves_procedure_for_a_new_researcher(learned_source_library):
    from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab

    config = ResearchConfig(observation_budget=256, compute_budget=180000, max_rounds=6,
        minimum_improvement=1e-7, seed=99534001)
    library = KnowledgeLibrary.from_dict(deepcopy(learned_source_library))
    prior_concepts = deepcopy([concept.to_dict() for concept in library.concepts])
    with AnonymousObjectLab(objects=4, readout="window_total_event", profile="confirmation",
            apparatus_seed=99531001, name_seed=99532001, wiring_seed=99533001,
            actuator_curvature=.31, relation_radius=.74, seed=99534001) as lab:
        agent = UnifiedResearchAgent(config, library=library, measurement_budget=3072)
        report = agent.run(lab)
    assert report["status"] == "unresolved" and report["concepts_added"] == []
    assert report["used_sources"] == []
    assert [concept.to_dict() for concept in library.concepts] == prior_concepts
    assert library.research_drafts
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    draft = restored.research_drafts[-1]
    assert draft["status"] == "unresolved" and not draft.get("resolved_concept_id")
    assert any(action["action"] == "preserve_research_draft" for action in report["actions"])

    # A distinct researcher receives measured roles, and pays to refit the
    # unfinished procedure. No source confirmation verdict is transferred.
    from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
    with UnifiedObjectLab(objects=4, readout="window_total_event", rename_prefix="later_",
                          relation_radius=.74, seed=131) as later:
        training = later.acquire(later.random_probe(457, length=24), ("fine",)).observations
        successor = UnifiedResearchAgent(config, library=restored, measurement_budget=3072)
        budget, actions = SearchBudget(8000), []
        claims = successor._program_claims(later.spec_for(("fine",)), training, budget, [], actions)
    assert any(item.get("research_draft_id") == draft["draft_id"] for item in claims)
    assert any(action["action"] == "replay_research_draft" for action in actions)
    assert 0 < budget.spent <= budget.limit
    assert not draft.get("resolved_concept_id")


def numeric_rows(*, width=3, threshold=.15, field="meter", control="x",
                 meter_scale=1., meter_offset=0., response_scale=1., length=48):
    rows, history, responses = [], [], []
    for index in range(length):
        reading = (-.9, .35, -.4, .85, -.15, .6, -.65, .1)[index % 8]
        command = ((index * 7) % 19 - 9) / 9.
        response = (1. + .7 * command + .3 * reading if reading >= threshold
                    else -.4 - .5 * command + .3 * reading)
        responses.append(response)
        target = response_scale * sum(responses[-width:])
        features = {control: command, field: meter_scale * reading + meter_offset}
        rows.append(Observation(features, target, tuple(history)))
        history.append({**features, "__target__": target})
    return rows


def numeric_procedure(*, decoded=False):
    from hegel_machine.cognition.conditional_coordinates import _masked
    from hegel_machine.cognition.observed_condition_uncertainty import assess_observed_condition_uncertainty
    from hegel_machine.cognition.research_programs import _Fitter, _entry

    rows = numeric_rows()
    condition = Program("ge", children=(Program("feature", name="meter"),
                                        Program("const", value=.225)))
    basis = [Program("feature", name="x"), Program("feature", name="meter"),
             Program("const", value=1.)]
    budget = SearchBudget(20000)
    fit = _Fitter(rows, budget, .5).fit([
        _masked(term, condition, side, 3, "inside_window")
        for term in basis for side in (True, False)])
    assert fit is not None and fit["loss"] < 1e-15
    step = {"operation": "discover_conditional_coordinates", "question": "Retest a paid condition",
            "program": fit["program"], "loss": fit["loss"],
            "gate_coordinate": Program("feature", name="meter").to_dict(),
            "threshold": .225, "threshold_interval": [.1, .35], "width": 3,
            "placement": "inside_window", "branch_basis": [term.to_dict() for term in basis[:-1]]}
    if decoded:
        step = {"operation": "discover_decoded_conditional_coordinates",
                "question": "Decode and refit the original response", "program": fit["program"],
                "loss": fit["loss"], "width": 3, "width_candidates": [3],
                "placement": "inside_window", "gate_names": ["meter"],
                "branch_basis": [term.to_dict() for term in basis[:-1]],
                "inner_selection_steps": [{**step, "program": fit["program"].to_dict()}]}
    selected = _entry(fit, [step], len(rows), budget.spent)
    measured = TaskSpec("numeric_source", "Paid auxiliary condition", ("x", "meter"), (), (),
                        feature_bounds={"x": (-1., 1.)}, max_probe_length=64)
    selected["condition_uncertainty"] = assess_observed_condition_uncertainty(
        measured, rows, selected, budget, tolerance=1e-7)
    assert selected["condition_uncertainty"]["status"] == "unresolved"
    return selected, measured, rows


def test_numeric_unresolved_question_survives_json_without_supported_concept():
    selected, measured, _ = numeric_procedure()
    library, actions = KnowledgeLibrary(), []
    record = remember_research_draft(library, selected, measured, actions)
    assert record is not None and not library.concepts
    assert record["source_schema"] is None
    assert record["control_features"] == ["x"] and record["observed_features"] == ["meter"]
    assert record["condition_uncertainty"] == selected["condition_uncertainty"]
    assert record["input_mapping"] == {"x": "x", "meter": "meter"}
    assert record["research_program"]["graph"] == selected["research_program"]["graph"]
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    assert restored.research_drafts == library.research_drafts
    assert not restored.concepts
    assert remember_research_draft(library, selected, measured, []) is record
    supported = {**selected, "condition_uncertainty": {**selected["condition_uncertainty"], "status": "supported"}}
    assert remember_research_draft(library, supported, measured, []) is None


@pytest.mark.parametrize("decoded", [False, True])
def test_numeric_draft_rebinds_and_refits_current_paid_original_response(decoded):
    selected, measured, old_rows = numeric_procedure(decoded=decoded)
    library = KnowledgeLibrary()
    original = remember_research_draft(library, selected, measured, [])
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    before = deepcopy(restored.to_dict())
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7), library=restored)
    new_spec = TaskSpec("new_numeric_mechanism", "Changed paid response", ("reading", "command"), (), (),
                        feature_bounds={"command": (-1., 1.)}, max_probe_length=64)
    training = numeric_rows(width=5, threshold=.55, field="reading", control="command",
                            meter_scale=2.3, meter_offset=-.4, response_scale=1.8)
    budget, actions = SearchBudget(180000), []
    claims = replay_research_drafts(agent, new_spec, training, budget, actions)
    assert claims and all(claim["research_draft_id"] == original["draft_id"] for claim in claims)
    current = min(claims, key=lambda item: item["training_loss"])
    assert current["training_loss"] == pytest.approx(mse(current["candidate"].program, training))
    assert current["candidate"].program.required_features() == {"reading", "command"}
    future = numeric_rows(width=5, threshold=.55, field="reading", control="command",
                          meter_scale=2.3, meter_offset=-.4, response_scale=1.8, length=61)
    decoded_steps = [step for step in current["research_program"]["selection_steps"]
                     if step["operation"] == "discover_decoded_conditional_coordinates"]
    if decoded or decoded_steps:
        assert current["training_loss"] < 1e-15
        assert any(step.get("width") == 5 for step in current["research_program"]["selection_steps"])
        assert mse(current["candidate"].program, future) < 1e-15
        if decoded_steps:
            assert all(set(step["gate_names"]) == {"reading"} for step in decoded_steps)
            assert all(step["history_bound"] == max(len(row.history) for row in training)
                       for step in decoded_steps)
    else:
        # Ordinary replay can reach a joint threshold/duration local minimum.
        # The draft proposes its honestly measured fit for current validation;
        # it transfers neither the old bracket nor a supported verdict.
        steps = [step for step in current["research_program"]["selection_steps"]
                 if step["operation"] == "discover_conditional_coordinates"]
        assert steps and all(step["gate_coordinate"]["name"] in ("reading", "command") for step in steps)
        assert all(step["history_bound"] == max(len(row.history) for row in training) for step in steps)
        assert any(len({trial["width"] for trial in step["parameter_trials"]}) > 1 for step in steps)
    assert restored.to_dict() == before and not restored.concepts
    assert all(action["status"] == "requires_current_validation" for action in actions)
    assert 0 < budget.spent <= budget.limit
    assert mse(selected["candidate"].program, old_rows) < 1e-15


def test_same_names_cannot_swap_a_command_and_a_paid_reading_during_replay():
    from hegel_machine.cognition.abstractions import AbstractTemplate
    from hegel_machine.cognition.research_drafts import _numeric_mappings

    selected, measured, _ = numeric_procedure()
    record = remember_research_draft(KnowledgeLibrary(), selected, measured, [])
    template = AbstractTemplate.from_dict(record["input_template"])
    # The field names recur with different public roles. An identity name
    # binding would make the old meter a command on this current apparatus.
    changed = TaskSpec("roles_changed", "", ("x", "meter"), (), (),
                       feature_bounds={"meter": (-1., 1.)})
    bindings = list(_numeric_mappings(record, template, changed, True))
    assert len(bindings) == 1
    assert dict(zip(template.source_features, (bindings[0][role] for role in template.roles))) == {
        "x": "meter", "meter": "x"}
    assert list(_numeric_mappings(record, template, changed, False)) == []
