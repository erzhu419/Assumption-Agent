"""Actual paired readings separate an input boundary error from a changed law."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.competing_object_experiments import _forecast, assess_competing_object_probe
from hegel_machine.cognition.instrument_models import fit_instrument_model, instrument_binding, rebind_instrument_model
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.relation_input_diagnosis import bind_relation_entry, measured_relation_entry
from hegel_machine.cognition.relation_revisions import propose_relation_revisions
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from hegel_machine.cognition.relocatable_programs import abstract_measurements
from hegel_machine.cognition.research_program_inputs import abstract_research_inputs
from hegel_machine.cognition.scale_bindings import bind_measurement_arguments
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


SETTINGS = dict(objects=4, apparatus_seed=9013, readout="window_total_event",
                relation_direction="right", relation_radius=.8)


def acquire(lab, probe):
    batch = lab.acquire(probe, ("fine",))
    history, pairs = [], []
    for point, measured in zip(probe.sequence, batch.observations):
        pairs.append((Observation(point, measured.target, tuple(history)), measured))
        history.append({**point, "__target__": measured.target})
    return pairs, batch


def edge_probe(name, *, focus=1.5):
    point = {"q_o0": .2, "q_o1": .7, "q_o2": 0., "q_o3": 0.,
             "x_o0": 0., "x_o1": 1.2, "x_o2": 2., "x_o3": -2.,
             "u_o0": 0., "u_o1": 0., "u_o2": 0., "u_o3": 0., "focus": focus}
    return Probe(name, tuple({**point, "x_o1": distance} for distance in (1.2, .4, .4, .4, .4, 1.2, 1.2, 1.2, 1.2)))


def law(spec, coupling):
    schema = json.loads(spec.context["object_schema"])
    f = lambda name: Program("feature", name=name)
    c = lambda value: Program("const", value=value)
    fields = [schema["fields"][name] for name in schema["objects"]]
    predicate = Program("lt", children=(Program("arg", name="edge_source_quantity"),
                                         Program("arg", name="edge_target_quantity")))
    flow = Program("mul", children=(c(coupling), Program("relation_sum", name="right", children=(
        c(.8), predicate, *(f(item[role]) for item in fields for role in ("f", "x", "q"))))))
    total = Program("object_sum", children=tuple(f(item[role]) for item in fields for role in ("q", "u")))
    response = Program("add", children=(total, Program("mul", children=(c(.7),
        Program("event_delta", lag=1, children=(flow,))))))
    return Program("window_sum", lag=3, children=(response,))


def plan(probe, program, binding):
    executable = bind_measurement_arguments(program, binding)
    return {"probe": probe, "predicted_fields": {name: _forecast(mapping, probe.sequence)
        for name, mapping in binding.items()}, "hypotheses": [{"hypothesis_id": "retained_law",
        "candidate_id": "retained_law", "program": program.to_dict(),
        "predictions": _forecast(executable, probe.sequence)}]}


def binding_error(binding, pairs):
    return max((program.predict(command) - measured.features[name]) ** 2
               for command, measured in pairs for name, program in binding.items())


def legacy_source_model(fitted):
    """Restore a pre-0.91 single-explanation source from genuinely paid fitting."""
    stored = deepcopy(fitted)
    for field in stored["fields"].values():
        # Historical files stored only the selected program. Revised models
        # retain their competing hypotheses and must purchase distinguishing data.
        field.pop("hypotheses", None)
    return json.loads(json.dumps(stored))


def resolve_current_model(lab, model, budget):
    """Use the actual controller and acquired readings to settle revised inputs."""
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7))
    agent._learn_instruments = agent._revise_inputs = True
    model["model_id"] = "revised_instrument"
    batches, actions = [], []

    def purchase(channels, purpose, *, probe):
        assert channels == ("fine",)
        assert purpose == "distinguish_instrument_hypotheses"
        pairs, batch = acquire(lab, probe)
        batches.append(pairs)
        return batch.observations

    resolved = agent._distinguish_instruments(lab, model, ("fine",), purchase,
        lambda channels: batches, budget, actions)
    assert resolved is model
    assert all(len([item for item in field["hypotheses"] if item["status"] != "refuted"]) == 1
               for field in model["fields"].values())
    return sum(map(len, batches))


def test_purchased_same_trajectory_repairs_overlapping_selector_cuts_and_rebinds_after_json():
    with UnifiedObjectLab(**SETTINGS, relation_condition="source_less") as lab:
        spec = lab.spec_for(("fine",))
        random_pairs, _ = acquire(lab, lab.random_probe(183, length=48))
        budget = SearchBudget(30000)
        original = legacy_source_model(fit_instrument_model(lab.spec, spec, random_pairs, budget))
        old = instrument_binding(original)
        old_payload = deepcopy(original)
        probe = edge_probe("existing_counterexample")
        pairs, batch = acquire(lab, probe)
        # Derive the response coefficient from a paid transition; no private
        # apparatus coefficient enters instrument fitting or diagnosis.
        transition = batch.transitions[1]
        before, after = transition["before"], transition["after"]
        coupling = (before["measured_q_o0"] - after["measured_q_o0"]) / before["measured_q_o0"]
        program = law(spec, coupling)
        point = pairs[1][0]
        assert sum(old[f"measured_f_o{index}"].predict(point) for index in range(4)) == 2.
        assessment = assess_competing_object_probe(plan(probe, program, old), pairs, budget)
        assert assessment["calibration_status"] == "counterexample"
        assert assessment["field_losses"]["measured_f_o1"] == 1.
        assert assessment["hypotheses"][0]["measured_loss"] < 1e-20
        assert assessment["hypotheses"][0]["forecast_loss"] > 1e-5

        repaired = fit_instrument_model(lab.spec, spec, [*random_pairs, *pairs], budget)
        assert repaired["status"] == "requires_fresh_validation"
        extra_rows = resolve_current_model(lab, repaired, budget)
        binding = instrument_binding(repaired)
        assert set(binding) == set(spec.feature_names)
        assert binding_error(binding, pairs) < 1e-10
        assert sum(binding[f"measured_f_o{index}"].predict(point) for index in range(4)) == 1.
        fresh, _ = acquire(lab, lab.random_probe(885, length=32))
        assert binding_error(binding, fresh) < 1e-10
        assert original == old_payload
        assert repaired["status"] == "supported_on_measured_trajectory"
        assert 0 < budget.spent <= budget.limit
        assert lab.costs["observation_cost"] == 89 + extra_rows
        assert lab.costs["measurement_cost"] == 12 * (89 + extra_rows)

    saved = json.loads(json.dumps(repaired))
    with UnifiedObjectLab(**SETTINGS, relation_condition="source_less", rename_prefix="later_") as renamed:
        rebound = rebind_instrument_model(saved, renamed.spec_for(("fine",)))
        pairs, _ = acquire(renamed, edge_probe("new_context_same_boundary"))
        assert set(rebound) == set(renamed.spec_for(("fine",)).feature_names)
        assert binding_error(rebound, pairs) < 1e-10
        corrected = assess_competing_object_probe(plan(edge_probe("new_context_same_boundary"),
            law(renamed.spec_for(("fine",)), coupling), rebound), pairs, SearchBudget(5000))
        assert corrected["calibration_status"] == "supported_on_probe"
        assert corrected["hypotheses"][0]["forecast_loss"] < 1e-10


def test_changed_relation_refutes_the_response_without_refuting_correct_input_calibration():
    with UnifiedObjectLab(**SETTINGS, relation_condition="source_less") as source, \
         UnifiedObjectLab(**SETTINGS, relation_condition="source_greater") as changed:
        spec = source.spec_for(("fine",))
        pairs, _ = acquire(source, source.random_probe(183, length=48))
        probe = edge_probe("mechanism_only", focus=1.)
        source_pairs, batch = acquire(source, probe)
        transition = batch.transitions[1]
        coupling = (transition["before"]["measured_q_o0"] - transition["after"]["measured_q_o0"]) / \
            transition["before"]["measured_q_o0"]
        fitted = fit_instrument_model(source.spec, spec, [*pairs, *source_pairs], SearchBudget(20000))
        model = legacy_source_model(fitted)
        original = deepcopy(model)
        binding = instrument_binding(model)
        current, _ = acquire(changed, probe)
        assert [row.features for _, row in current] == [row.features for _, row in source_pairs]
        assert max(abs(a.target - b.target) for (_, a), (_, b) in zip(source_pairs, current)) > .01
        assessment = assess_competing_object_probe(plan(probe, law(spec, coupling), binding),
                                                   current, SearchBudget(5000))
        assert assessment["calibration_status"] == "supported_on_probe"
        assert assessment["hypotheses"][0]["measured_loss"] > 1e-5
        assert assessment["hypotheses"][0]["forecast_loss"] > 1e-5
        repeated = fit_instrument_model(changed.spec, changed.spec_for(("fine",)),
            [*pairs, *current], SearchBudget(20000))
        # The changed response does not change any instrument hypothesis fitted
        # to the identical paid fields; new ambiguity is not force-bound.
        assert repeated["fields"] == fitted["fields"]
        assert model == original


def inherited_boundary_case(lab):
    """Learn a real measured source, then purchase its faulty selector endpoint."""
    spec = lab.spec_for(("fine",))
    random_pairs, _ = acquire(lab, lab.random_probe(183, length=48))
    old_model = legacy_source_model(fit_instrument_model(lab.spec, spec, random_pairs, SearchBudget(20000)))
    probe = edge_probe("paid_selector_endpoint")
    # Keep these already-paid positions outside the open radius interval.
    # Changing the selector calibration therefore must not resolve that interval.
    probe = replace(probe, sequence=tuple({**point, "x_o2": -2.} for point in probe.sequence))
    paired, batch = acquire(lab, probe)
    before, after = batch.transitions[1]["before"], batch.transitions[1]["after"]
    coupling = (after["measured_q_o0"] - before["measured_q_o0"]) / (
        before["measured_q_o1"] - before["measured_q_o0"])
    schema = json.loads(spec.context["object_schema"])
    fields = [schema["fields"][name] for name in schema["objects"]]
    f = lambda name: Program("feature", name=name)
    flow = Program("mul", children=(Program("const", value=coupling), Program("neighbor_sum",
        children=(Program("const", value=.8),
                  *(f(item[role]) for item in fields for role in ("f", "x", "q"))))))
    total = Program("object_sum", children=tuple(f(item[role]) for item in fields for role in ("q", "u")))
    measured = Program("window_sum", lag=3, children=(Program("add", children=(total,
        Program("mul", children=(Program("const", value=.7),
                                 Program("event_delta", lag=1, children=(flow,)))))),))
    assert max(abs(measured.predict(row) - row.target) for _, row in random_pairs + paired) < 1e-12
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7), library=KnowledgeLibrary())
    agent._learn_instruments = agent._revise_inputs = True
    agent.library.instrument_models.append(old_model)
    source, _ = agent.library.learn(Candidate("measured_window", measured, "readout"),
                                    spec, [row for _, row in random_pairs])
    agent.library.meta_priors.append({"source_concept_id": source, "object_schema": schema,
        "source_features": list(spec.feature_names), "control_features": list(lab.spec.feature_bounds),
        "structure": {"radius_interval": [.7, .9], "fits": [{"radius": .8,
            "family": "event_change", "radius_interval": [.7, .9]}]}})
    call = next(call for call in agent._recall(lab.spec) if call.definition.source_concept_id == source)
    repaired = fit_instrument_model(lab.spec, spec, [*random_pairs, *paired], SearchBudget(20000))
    resolve_current_model(lab, repaired, SearchBudget(20000))
    binding = instrument_binding(repaired)
    old = instrument_binding(old_model)
    assert old["measured_f_o1"] != binding["measured_f_o1"]
    assert old["measured_f_o1"].predict(paired[1][0]) == 1.
    assert binding["measured_f_o1"].predict(paired[1][0]) == 0.
    return agent, call, binding, paired


def assert_rebound_call_inherits_interval(agent, lab, call, binding, paired, expected_interval=(.7, .9)):
    before = deepcopy(agent.library.to_dict())
    entry = {"candidate": Candidate("retained_call", call, "call")}
    prepared = measured_relation_entry(entry, lab.spec_for(("fine",)), agent.library)
    assert prepared is not None and prepared[2] is None
    rebound = bind_relation_entry(prepared, binding)
    assert "research_program" not in rebound
    current = rebound["candidate"].program
    assert max(abs(current.predict(command) - row.target) for command, row in paired) < 1e-8
    assert max(abs(call.predict(command) - row.target) for command, row in paired) > .01
    research = RelationIntervalResearch(current, agent.library.meta_priors, SearchBudget(20000))
    assert research.intervals and all(item["radius_interval"] == list(expected_interval)
                                     for item in research.intervals)
    alternatives = research.alternatives([command for command, _ in paired], SearchBudget(30000))
    assert len(alternatives) >= 2
    # A different actual trajectory supplies a position inside the inherited
    # interval. Its complete event/window predictions still distinguish radii.
    probe = edge_probe("remaining_radius_question", focus=1.)
    probe = replace(probe, sequence=tuple({**point, "x_o2": -2.,
        "x_o1": sum(expected_interval) / 2 if index in (1, 2, 3, 4) else 1.2}
        for index, point in enumerate(probe.sequence)))
    future, _ = acquire(lab, probe)
    forecasts = [[item["candidate"].program.predict(command) for command, _ in future]
                 for item in alternatives]
    assert max(max(values) - min(values) for values in zip(*forecasts)) > .01
    assert agent.library.to_dict() == before
    return alternatives


def test_selector_repair_inherits_structure_interval_from_a_bare_measured_source_call():
    with UnifiedObjectLab(**{**SETTINGS, "relation_direction": "symmetric"}) as lab:
        agent, call, binding, paired = inherited_boundary_case(lab)
        assert_rebound_call_inherits_interval(agent, lab, call, binding, paired)


@pytest.mark.parametrize("save_command_inputs", [False, True], ids=["direct_measured_source", "saved_input_source"])
def test_selector_repair_inherits_saved_input_interval_from_a_bare_recalled_call_after_json(save_command_inputs):
    with UnifiedObjectLab(**{**SETTINGS, "relation_direction": "symmetric"}) as source_lab, \
         UnifiedObjectLab(**SETTINGS, relation_condition="source_less") as lab:
        agent, call, binding, _ = inherited_boundary_case(source_lab)
        # A command-space composite keeps its actual measured-input signature
        # before it becomes the anchor of a conditional local replacement.
        if save_command_inputs:
            source_pairs, _ = acquire(source_lab, source_lab.random_probe(885, length=24))
            source_rows = [command for command, _ in source_pairs]
            command_source, _ = agent.library.learn(Candidate("command_window", call, "call"),
                                                    source_lab.spec, source_rows)
            command_prior = {"source_concept_id": command_source, "object_schema": None,
                "source_features": list(source_lab.spec.feature_names),
                "control_features": list(source_lab.spec.feature_bounds)}
            agent.library.meta_priors.append(command_prior)
            agent._remember_inputs(command_prior, call, source_lab.spec_for(("fine",)),
                                   source_lab.spec, source_rows, SearchBudget(2000), [])
            assert command_prior.get("input_program")
            call = next(call for call in agent._recall(source_lab.spec)
                        if call.definition.source_concept_id == command_source)
        # First learn the actual conditional replacement, then save a live
        # parameter template. The eventual entry contains only its recalled call.
        training, _ = acquire(lab, lab.random_probe(183, length=48))
        revisions = propose_relation_revisions(lab.spec, [command for command, _ in training],
                                               SearchBudget(80000), [call])
        revised = min(revisions, key=lambda item: item["training_loss"])
        assert revised["training_loss"] < 1e-7
        assert revised["research_program"]["selection_steps"][-1]["compact_replacement"]
        valid_probe = edge_probe("source_interval_evidence", focus=1.)
        valid_probe = replace(valid_probe, sequence=tuple({**point, "x_o2": -2.}
                                                       for point in valid_probe.sequence))
        valid, _ = acquire(lab, valid_probe)
        rows = [command for command, _ in valid]
        interval = RelationIntervalResearch(revised["candidate"].program, agent.library.meta_priors,
            SearchBudget(20000), research_program=revised["research_program"])
        expected_interval = interval.intervals[0]["radius_interval"]
        selected = interval.alternatives(rows, SearchBudget(30000))[0]
        measured_procedure = abstract_research_inputs(selected["research_program"],
            lab.spec_for(("fine",)), agent.library)
        assert measured_procedure is not None
        parameter_template = Program.from_dict(measured_procedure["selection_steps"][-1]["parameter_template"])
        assert parameter_template.required_features() <= set(lab.spec_for(("fine",)).feature_names)
        source, _ = agent.library.learn(selected["candidate"], lab.spec, rows)
        prior = {"source_concept_id": source, "research_program": selected["research_program"],
                 "source_features": list(lab.spec.feature_names),
                 "control_features": list(lab.spec.feature_bounds), "object_schema": None}
        agent.library.meta_priors.append(prior)
        agent._remember_inputs(prior, selected["candidate"].program, lab.spec_for(("fine",)),
                               lab.spec, rows, SearchBudget(2000), [])
        assert prior.get("input_program") and prior.get("input_research_program")
        library = KnowledgeLibrary.from_dict(json.loads(json.dumps(agent.library.to_dict())))
        restored = UnifiedResearchAgent(agent.config, library=library)
        restored._learn_instruments = restored._revise_inputs = True
        call = next(call for call in restored._recall(lab.spec)
                    if call.definition.source_concept_id == source)
        assert call.name == prior["input_program"]["definition"]["definition_id"]
        endpoint = edge_probe("conditional_selector_endpoint")
        endpoint = replace(endpoint, sequence=tuple({**point, "x_o2": -2.}
                                                   for point in endpoint.sequence))
        paired, _ = acquire(lab, endpoint)
        assert_rebound_call_inherits_interval(restored, lab, call, binding, paired, expected_interval)

        if not save_command_inputs:
            from hegel_machine.cognition.relation_revisions import _replacement_call
            original_call = revised["candidate"].program
            # An actual replacement that adds a raw control changes the input
            # contract; equal arity or the same source alone cannot recover it.
            extra = _replacement_call(original_call, Program("add", children=(
                original_call.expanded(), Program("feature", name="focus"))), "extra_control")
            abstraction = abstract_measurements(extra, lab.spec_for(("fine",)), agent.library)
            assert not abstraction["complete"]
            assert any(item["reason"] == "source_argument_roles_unavailable"
                       for item in abstraction["unresolved_arguments"])
