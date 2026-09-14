"""Changing a measured input must not discard the learned temporal procedure."""

from copy import deepcopy
from dataclasses import replace
import json

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget
from hegel_machine.cognition.research_program_inputs import abstract_research_inputs
from hegel_machine.cognition.research_programs import propose_research_programs, replay_research_program
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def pairs(lab, seed):
    probe = lab.random_probe(seed, length=32)
    measured = lab.acquire(probe, ("fine",)).observations
    history, result = [], []
    for point, row in zip(probe.sequence, measured):
        result.append((Observation(dict(point), row.target, tuple(history)), row))
        history.append(dict(point))
    return result


def observed_readout(lab, gain):
    schema = json.loads(lab.spec_for(("fine",)).context["object_schema"])
    quantities = tuple(Program("feature", name=schema["fields"][obj]["q"])
                       for obj in schema["objects"])
    total = Program("object_sum", children=quantities)
    impulse = Program("feature", name=schema["fields"][schema["objects"][0]]["u"])
    event = Program("event_delta", lag=1, children=(impulse,))
    combined = Program("add", children=(
        Program("mul", children=(Program("const", value=gain), total)),
        Program("mul", children=(Program("const", value=-.7), event))))
    return total, event, Program("window_sum", lag=3, children=(combined,))


def commands_for(paid, readout):
    return [replace(command, target=readout.predict(measured)) for command, measured in paid]


def mse(program, rows):
    return sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows)


def learned_procedure(lab):
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7, max_candidates=24),
                                  library=KnowledgeLibrary())
    agent._learn_instruments = agent._revise_inputs = True
    fine = lab.spec_for(("fine",))
    batches = [pairs(lab, seed) for seed in (183, 885)]
    agent._fit_instruments(lab.spec, fine, batches, SearchBudget(24000), [])
    total, event, readout = observed_readout(lab, 1.3)
    for index, program in enumerate((total, event)):
        evidence = [replace(row, target=program.predict(row)) for _, row in batches[0]]
        concept, _ = agent.library.learn(Candidate(f"component:{index}", program, "readout"), fine, evidence)
        agent.library.meta_priors.append({"source_concept_id": concept,
            "source_features": list(fine.feature_names), "control_features": list(fine.feature_bounds),
            "object_schema": json.loads(fine.context["object_schema"])})
    rows = commands_for([pair for batch in batches for pair in batch], readout)
    proposals = propose_research_programs(lab.spec, rows, SearchBudget(60000), agent._recall(lab.spec))
    selected = min(proposals, key=lambda item: item["training_loss"])
    assert selected["training_loss"] < 1e-10
    return agent, rows, selected


def test_saved_temporal_procedure_rebinds_revised_calibration_after_restore():
    with UnifiedObjectLab(objects=2, apparatus_seed=9013, actuator_curvature=.12) as old_lab:
        agent, old_rows, selected = learned_procedure(old_lab)
        original = deepcopy(selected["research_program"])
        converted = abstract_research_inputs(original, old_lab.spec_for(("fine",)), agent.library)
        assert converted is not None
        assert selected["research_program"] == original
        assert any(step["operation"] == "transform_component" for step in converted["selection_steps"])

        concept, _ = agent.library.learn(selected["candidate"], old_lab.spec, old_rows)
        prior = {"source_concept_id": concept, "research_program": original,
                 "source_features": list(old_lab.spec.feature_names),
                 "control_features": list(old_lab.spec.feature_bounds), "object_schema": None}
        agent.library.meta_priors.append(prior)
        agent._remember_inputs(prior, selected["candidate"].program, old_lab.spec_for(("fine",)),
                               old_lab.spec, old_rows, SearchBudget(1000), [])
        assert prior["input_research_program"] == converted
        library = KnowledgeLibrary.from_dict(json.loads(json.dumps(agent.library.to_dict())))

    with UnifiedObjectLab(objects=2, apparatus_seed=9013, actuator_curvature=.39,
                          rename_prefix="changed_") as changed_lab:
        learner = UnifiedResearchAgent(agent.config, library=library)
        learner._learn_instruments = learner._revise_inputs = True
        new_batches = [pairs(changed_lab, seed) for seed in (1831, 8851)]
        actions = []
        learner._fit_instruments(changed_lab.spec, changed_lab.spec_for(("fine",)),
                                 new_batches, SearchBudget(24000), actions)
        assert any(item["action"] == "instrument_mapping_counterexample" for item in actions)
        readout = observed_readout(changed_lab, -.5)[2]
        training = commands_for([pair for batch in new_batches for pair in batch], readout)
        future = commands_for(pairs(changed_lab, 1981), readout)

        # The old procedure can change its linear head, but its embedded cubic
        # calibration cannot explain the new measured trajectory by that refit.
        rename = dict(zip(old_lab.spec.feature_names, changed_lab.spec.feature_names))
        stale = replay_research_program(original, training, SearchBudget(2000), feature_mapping=rename)
        assert stale is not None and mse(stale["candidate"].program, future) > 1e-4

        claims = learner._program_claims(changed_lab.spec, training, SearchBudget(4000),
                                          learner._recall(changed_lab.spec), [])
        replayed = [claim for claim in claims if concept in claim.get("method_source_ids", [])]
        assert replayed
        assert min(mse(claim["candidate"].program, future) for claim in replayed) < 1e-9
        assert all(claim["research_program"]["reused_procedure"] for claim in replayed)
        assert all(claim["candidate"].program.required_features() <= set(changed_lab.spec.feature_names)
                   for claim in replayed)
        assert prior["research_program"] == original


def test_unrecovered_command_step_does_not_create_a_partly_rebound_procedure():
    with UnifiedObjectLab(objects=2, apparatus_seed=9013, actuator_curvature=.12) as lab:
        agent, _, selected = learned_procedure(lab)
        original = deepcopy(selected["research_program"])
        unresolved = deepcopy(original)
        unresolved["selection_steps"][0]["program"] = Program("feature", name=lab.spec.feature_names[0]).to_dict()
        assert abstract_research_inputs(unresolved, lab.spec_for(("fine",)), agent.library) is None
        assert selected["research_program"] == original


def test_unresolved_parameter_template_rebinds_with_the_measured_procedure():
    from hegel_machine.cognition.scale_bindings import bind_measurement_arguments

    with UnifiedObjectLab(objects=2, apparatus_seed=9013, actuator_curvature=.12) as old_lab:
        agent, _, selected = learned_procedure(old_lab)
        source = selected["candidate"].program
        procedure = deepcopy(selected["research_program"])
        template = Program("mul", children=(Program("arg", name="relation_parameter_0"), source))
        procedure["selection_steps"].append({
            "operation": "resolve_relation_intervals", "program": source.to_dict(),
            "parameters": [{"symbol": "relation_parameter_0", "radius": 1.,
                            "radius_intervals": [[.5, 1.5]]}],
            "parameter_template": template.to_dict()})
        original = deepcopy(procedure)
        converted = abstract_research_inputs(procedure, old_lab.spec_for(("fine",)), agent.library)
        assert converted is not None and procedure == original
        saved = json.loads(json.dumps(converted))
        parameter = saved["selection_steps"][-1]
        assert parameter["parameters"] == original["selection_steps"][-1]["parameters"]
        measured_template = Program.from_dict(parameter["parameter_template"])
        assert measured_template.required_features() <= set(saved["input_features"])
        assert measured_template.children[0] == Program("arg", name="relation_parameter_0")

    with UnifiedObjectLab(objects=2, apparatus_seed=9013, actuator_curvature=.39) as changed_lab:
        measured_spec = changed_lab.spec_for(("fine",))
        new_batches = [pairs(changed_lab, seed) for seed in (1831, 8851)]
        agent._fit_instruments(changed_lab.spec, measured_spec, new_batches, SearchBudget(24000), [])
        binding = agent._instrument_binding(measured_spec, changed_lab.spec)
        rebound = bind_measurement_arguments(measured_template, binding)
        # A still-open relation parameter does not capture either the old
        # calibration or the lexical inputs of its component calls.
        def choose(program):
            if program.op == "arg" and program.name == "relation_parameter_0":
                return Program("const", value=.8)
            return replace(program, children=tuple(choose(child) for child in program.children))
        rebound, measured = choose(rebound), choose(measured_template)
        fresh = pairs(changed_lab, 1981)
        assert max(abs(rebound.predict(command) - measured.predict(row))
                   for command, row in fresh) < 1e-7
        stale = choose(template)
        assert max(abs(stale.predict(command) - measured.predict(row))
                   for command, row in fresh) > 1e-4


def test_unknown_parameter_input_does_not_leave_a_stale_template_in_a_converted_procedure():
    with UnifiedObjectLab(objects=2, apparatus_seed=9013, actuator_curvature=.12) as lab:
        agent, _, selected = learned_procedure(lab)
        procedure = deepcopy(selected["research_program"])
        procedure["selection_steps"].append({
            "operation": "resolve_relation_intervals", "program": selected["candidate"].program.to_dict(),
            "parameters": [{"symbol": "relation_parameter_0"}],
            "parameter_template": Program("mul", children=(
                Program("arg", name="relation_parameter_0"),
                Program("feature", name=lab.spec.feature_names[0]))).to_dict()})
        assert abstract_research_inputs(procedure, lab.spec_for(("fine",)), agent.library) is None
