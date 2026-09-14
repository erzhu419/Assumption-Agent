"""Real learned higher-order programs survive replacement of their calibration."""

import json
from pathlib import Path

import pytest

from hegel_machine.cognition.instrument_models import fit_instrument_model, rebind_instrument_model
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget
from hegel_machine.cognition.relocatable_programs import abstract_measurements, instantiate_measurements
from hegel_machine.cognition.scale_bindings import bind_measurement_arguments
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab


@pytest.fixture(scope="module")
def learned():
    return json.loads((Path(__file__).parent / "fixtures/cognition_v063_relocatable_programs.json").read_text())


def pairs(lab, seed, length=48):
    probe = lab.random_probe(seed, length=length)
    measured = lab.observe(probe, ("fine",))
    history, result = [], []
    for point, row in zip(probe.sequence, measured):
        result.append((Observation(point, row.target, tuple(history)), row))
        history.append(dict(point))
    return result


@pytest.mark.parametrize("position,readout", [(3, "total_event"), (4, "filtered_total"),
                                              (5, "window_total_event"), (7, "window_total_event")])
def test_v063_program_preserves_measurement_calls_and_accepts_new_calibration(learned, position, readout):
    library = KnowledgeLibrary.from_dict(learned["knowledge"])
    original = Program.from_dict(learned["programs"][str(position)])
    frozen = original.to_dict()
    with UnifiedObjectLab(apparatus_seed=6131, readout=readout,
                          rename_prefix="new_", actuator_curvature=.43) as changed:
        spec = changed.spec_for(("fine",))
        abstraction = abstract_measurements(original, spec, library)
        assert abstraction["complete"], abstraction
        assert abstraction["restored_calls"]
        assert set(abstraction["required_fields"]) <= set(spec.feature_names)
        if position == 7:
            assert "concept_0006" in abstraction["expanded_sources"]
            assert "concept_0004" in {item["source_id"] for item in abstraction["restored_calls"]}
        fresh = [pair for seed in (761, 2941) for pair in pairs(changed, seed, length=64)]
        old_binding = rebind_instrument_model(library.instrument_models[1], spec)
        roundtrip = instantiate_measurements(abstraction, old_binding)
        assert max(abs(original.predict(command) - roundtrip.predict(command))
                   for command, _ in fresh) < 1e-12
        calibration = fit_instrument_model(changed.spec, spec, pairs(changed, 553), SearchBudget(12000))
        restored = json.loads(json.dumps(abstraction))
        rebound = instantiate_measurements(restored, rebind_instrument_model(calibration, spec))
        previous_loss = sum((original.predict(command) - row.target) ** 2 for command, row in fresh) / len(fresh)
        new_loss = sum((rebound.predict(command) - row.target) ** 2 for command, row in fresh) / len(fresh)
        assert previous_loss > 1e-4
        assert new_loss < 1e-7
    assert original.to_dict() == frozen


def test_extra_raw_command_dependency_is_preserved_and_conversion_stays_incomplete(learned):
    library = KnowledgeLibrary.from_dict(learned["knowledge"])
    original = Program.from_dict(learned["programs"]["7"])
    program = Program("add", children=(original, Program("feature", name="q_o0")))
    with UnifiedObjectLab(apparatus_seed=6131) as lab:
        abstraction = abstract_measurements(program, lab.spec_for(("fine",)), library)
    assert not abstraction["complete"]
    assert abstraction["unresolved_commands"] == ["q_o0"]
    assert Program.from_dict(abstraction["program"]).children[1] == program.children[1]
    with pytest.raises(ValueError, match="incomplete"):
        instantiate_measurements(abstraction, {})


def test_calibrated_lag_arguments_survive_lexical_calls_and_temporal_composition(learned):
    library = KnowledgeLibrary.from_dict(learned["knowledge"])
    with UnifiedObjectLab(apparatus_seed=6131, actuator_curvature=.43) as lab:
        spec = lab.spec_for(("fine",))
        schema = json.loads(spec.context["object_schema"])
        name = schema["fields"]["o0"]["q"]
        raw = Program("if", children=(Program("available", lag=1),
            Program("lag", name=name, lag=1), Program("feature", name=name)))
        source_id, _ = library.learn(Candidate("measured_lag", raw, "history"), spec)
        library.meta_priors.append({"source_concept_id": source_id, "object_schema": schema})
        definition = library.definitions[f"{source_id}:abstract:0:definition"]
        binding = library.definition_bindings[definition.definition_id]
        template = next(item for item in library.abstracts if item.template_id == binding["template_id"])
        old = rebind_instrument_model(library.instrument_models[1], spec)
        role_binding = {role: old[field] for role, field in zip(template.roles, template.source_features)}
        call = Program("call", name=definition.definition_id, definition=definition,
            children=tuple(bind_measurement_arguments(argument, role_binding) for argument in binding["arguments"]))
        window = Program("window_sum", lag=3, children=(call,))
        state = Program("accumulate", value=.5, children=(window, Program("const", value=0.)))
        event = Program("event_delta", lag=1, children=(state,))
        outer = Definition("nested_control_program", ("input",),
                           Program("add", children=(Program("arg", name="input"), Program("const", value=0.))),
                           "command_parent")
        program = Program("call", name=outer.definition_id, definition=outer, children=(event,))
        abstract = abstract_measurements(program, spec, library)
        assert abstract["complete"], abstract
        calibration = fit_instrument_model(lab.spec, spec, pairs(lab, 557), SearchBudget(12000))
        rebound = instantiate_measurements(abstract, rebind_instrument_model(calibration, spec))
        recent, previous_q, previous_state, state_value = [], None, 0., 0.
        for index, (command, measured) in enumerate(pairs(lab, 994, length=64)):
            quantity = measured.features[name]
            recent.append(quantity if previous_q is None else previous_q)
            state_value = .5 * state_value + sum(recent[-3:])
            expected = 0. if index == 0 else state_value - previous_state
            assert rebound.predict(command) == pytest.approx(expected, abs=1e-8)
            previous_q, previous_state = quantity, state_value
