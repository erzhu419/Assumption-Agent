"""Relocatable measurement programs preserve the lexical scope of edge predicates."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget
from hegel_machine.cognition.relocatable_programs import abstract_measurements, instantiate_measurements
from hegel_machine.cognition.scale_bindings import bind_measurement_arguments
from hegel_machine.cognition.research_programs import _entry
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab


def arg(name):
    return Program("arg", name=name)


def relation(spec, predicate, radius=None):
    schema = json.loads(spec.context["object_schema"])
    return Program("relation_sum", name="right", children=(radius or Program("const", value=.8), predicate,
        *(Program("feature", name=schema["fields"][group][role])
          for group in schema["objects"] for role in ("f", "x", "q"))))


def call(name, arguments, body, children):
    definition = Definition(name, tuple(arguments), body, name)
    return Program("call", name=name, definition=definition, children=tuple(children))


def temporal(program):
    return Program("window_sum", lag=3, children=(Program("event_delta", lag=1, children=(program,)),))


def assert_roundtrip(program, spec, rows):
    frozen = program.to_dict()
    payload = abstract_measurements(program, spec, KnowledgeLibrary())
    assert payload["complete"], payload["unresolved_arguments"]
    assert payload["unresolved_arguments"] == []
    assert payload["unresolved_commands"] == []
    restored = json.loads(json.dumps(payload))
    abstract = Program.from_dict(restored["program"])
    mapping = {name: Program("feature", name="renamed_" + name) for name in payload["required_fields"]}
    instantiated = instantiate_measurements(restored, mapping)
    for row in rows:
        renamed = replace(row, features={"renamed_" + name: value for name, value in row.features.items()},
            history=tuple({"renamed_" + name: value for name, value in past.items()} for past in row.history))
        assert abstract.predict(row) == pytest.approx(program.predict(row), abs=1e-12)
        assert instantiated.predict(renamed) == pytest.approx(program.predict(row), abs=1e-12)
    assert program.to_dict() == frozen
    return abstract


def test_measured_relation_predicate_abstracts_and_instantiates_with_real_object_fields():
    with UnifiedObjectLab(objects=3, apparatus_seed=3019) as lab:
        spec = lab.spec_for(("fine",))
        rows = [row for index, length in enumerate((1, 2, 13))
                for row in lab.observe(lab.random_probe(3023 + index, length=length), ("fine",))]
        predicate = Program("lt", children=(arg("edge_target_quantity"), arg("edge_source_quantity")))
        program = temporal(relation(spec, predicate))
        assert any(abs(program.predict(row)) > 1e-5 for row in rows)
        assert_roundtrip(program, spec, rows)


def test_nested_calls_resolve_outer_arguments_without_capturing_edge_variables_of_the_same_name():
    with UnifiedObjectLab(objects=3, apparatus_seed=3037) as lab:
        spec = lab.spec_for(("fine",))
        rows = lab.observe(lab.random_probe(3041, length=17), ("fine",))
        compare = call("compare", ("edge_source_quantity", "cut"),
            Program("lt", children=(arg("cut"), arg("edge_source_quantity"))),
            (arg("edge_target_quantity"), arg("threshold")))
        predicate = Program("and", children=(
            Program("lt", children=(arg("edge_target_quantity"), arg("edge_source_quantity"))), compare))
        bounded = relation(spec, predicate, radius=arg("edge_source_quantity"))
        # Substitution crosses a nested call argument containing relation_sum.
        # The outer parameter of the same name supplies radius, never edge q.
        identity = call("identity", ("input",), arg("input"), (bounded,))
        program = call("outer", ("edge_source_quantity", "threshold"), temporal(identity),
                       (Program("const", value=.8), Program("const", value=-.1)))
        abstract = assert_roundtrip(program, spec, rows)
        expected = temporal(relation(spec, Program("and", children=(
            Program("lt", children=(arg("edge_target_quantity"), arg("edge_source_quantity"))),
            Program("lt", children=(Program("const", value=-.1), arg("edge_target_quantity")))))))
        assert any(abs(expected.predict(row)) > 1e-5 for row in rows)
        for row in rows:
            assert abstract.predict(row) == pytest.approx(expected.predict(row), abs=1e-12)


@pytest.mark.parametrize("name", ["external_parameter", "edge_source_quantity"])
def test_argument_outside_its_edge_scope_remains_unresolved(name):
    with UnifiedObjectLab(objects=3, apparatus_seed=3049) as lab:
        spec = lab.spec_for(("fine",))
        predicate = Program("lt", children=(arg("edge_target_quantity"), arg("edge_source_quantity")))
        program = Program("add", children=(relation(spec, predicate), arg(name)))
        payload = abstract_measurements(program, spec, KnowledgeLibrary())
        assert not payload["complete"]
        assert payload["unresolved_arguments"] == [{"argument": name, "reason": "unbound_lexical_argument"}]
        with pytest.raises(ValueError, match="incomplete"):
            instantiate_measurements(payload, {})


@pytest.fixture(scope="module")
def saved_input_contract():
    fixture = json.loads((Path(__file__).parent / "fixtures/cognition_v063_relocatable_programs.json").read_text())
    library = KnowledgeLibrary.from_dict(fixture["knowledge"])
    # This small historical fixture retains definitions and priors, not concepts.
    library.next_concept_index = 1 + max(int(prior["source_concept_id"].split("_")[-1])
                                        for prior in library.meta_priors)
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7), library=library)
    agent._learn_instruments = agent._revise_inputs = True
    original = Program.from_dict(fixture["programs"]["7"])
    with UnifiedObjectLab(apparatus_seed=6131) as lab:
        spec = lab.spec_for(("fine",))
        original = instantiate_measurements(abstract_measurements(original, spec, library),
                                           agent._instrument_binding(spec, lab.spec))
        probe = lab.random_probe(3083, length=24)
        measured = lab.observe(probe, ("fine",))
        commands = [Observation(point, row.target, tuple(probe.sequence[:index]))
                    for index, (point, row) in enumerate(zip(probe.sequence, measured))]
        source, _ = library.learn(Candidate("input_parent", original, "composition"), lab.spec, commands)
        prior = {"source_concept_id": source, "object_schema": None}
        library.meta_priors.append(prior)
        agent._remember_inputs(prior, original, spec, lab.spec, commands, SearchBudget(2000), [])
        assert prior.get("input_program")
        return json.loads(json.dumps(library.to_dict())), source, spec, lab.spec, commands, measured


def derived_condition(library, source, spec, command_spec, *, same_signature=True):
    inputs = next(prior for prior in library.meta_priors if prior["source_concept_id"] == source)["input_program"]
    signature = tuple(inputs["definition"]["arguments"])
    if not same_signature:
        signature = tuple("different_" + name for name in signature)
    original = dict(zip(inputs["template"]["roles"], inputs["template"]["source_features"]))
    fields = {original[item["name"]]: arg(name)
              for name, item in zip(signature, inputs["arguments"]) if item["op"] == "feature"}
    predicate = Program("lt", children=(arg("edge_target_quantity"), arg("edge_source_quantity")))
    measured = temporal(relation(spec, predicate))
    body = bind_measurement_arguments(measured, fields)
    agent = UnifiedResearchAgent(ResearchConfig(), library=library)
    agent._learn_instruments = True
    calibration = agent._instrument_binding(spec, command_spec)
    roles = {role: calibration[field] for role, field in original.items()}
    arguments = tuple(bind_measurement_arguments(Program.from_dict(item), roles) for item in inputs["arguments"])
    definition = Definition("derived_relation_condition", signature, body, source)
    return Program("call", name=definition.definition_id, definition=definition, children=arguments), measured, calibration


def test_derived_condition_recovers_saved_input_contract_and_preserves_research_after_json(saved_input_contract):
    saved, source, spec, command_spec, commands, measured_rows = saved_input_contract
    library = KnowledgeLibrary.from_dict(saved)
    prior_before = json.loads(json.dumps(next(p for p in library.meta_priors if p["source_concept_id"] == source)))
    derived, expected, calibration = derived_condition(library, source, spec, command_spec)
    assert derived.definition.definition_id not in library.definition_bindings
    abstraction = abstract_measurements(derived, spec, library)
    assert abstraction["complete"], abstraction["unresolved_arguments"]
    assert abstraction["restored_calls"][0]["definition_id"] == derived.definition.definition_id
    abstract = Program.from_dict(abstraction["program"])
    assert abstract.definition.body == derived.definition.body
    rebound = instantiate_measurements(json.loads(json.dumps(abstraction)), calibration)
    rename = {name: Program("feature", name="new_" + name) for name in abstraction["required_fields"]}
    renamed = instantiate_measurements(abstraction, rename)
    assert any(abs(expected.predict(row)) > 1e-5 for row in measured_rows)
    for command, measured in zip(commands, measured_rows):
        assert rebound.predict(command) == pytest.approx(derived.predict(command), abs=1e-12)
        assert abstract.predict(measured) == pytest.approx(expected.predict(measured), abs=1e-12)
        changed = replace(measured, features={"new_" + k: v for k, v in measured.features.items()},
            history=tuple({"new_" + k: v for k, v in row.items()} for row in measured.history))
        assert renamed.predict(changed) == pytest.approx(expected.predict(measured), abs=1e-12)

    fit = {"program": derived, "terms": [derived], "fit_terms": [derived], "coefficients": [0., 1.], "loss": 0.}
    selected = _entry(fit, [{"operation": "revise_relation_condition", "question": "Retest the changed relation",
                            "program": derived, "loss": 0.}], len(commands), 0)
    new_source, _ = library.learn(selected["candidate"], command_spec, commands)
    new_prior = {"source_concept_id": new_source, "object_schema": None,
                 "research_program": selected["research_program"]}
    library.meta_priors.append(new_prior)
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7), library=library)
    agent._learn_instruments = agent._revise_inputs = True
    agent._remember_inputs(new_prior, derived, spec, command_spec, commands, SearchBudget(2000), [])
    assert new_prior.get("input_program") and new_prior.get("input_research_program")
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    saved_new = next(p for p in restored.meta_priors if p["source_concept_id"] == new_source)
    answer = Program.from_dict(saved_new["input_research_program"]["selection_steps"][0]["program"])
    for row in measured_rows:
        assert answer.predict(row) == pytest.approx(expected.predict(row), abs=1e-12)
    assert next(p for p in library.meta_priors if p["source_concept_id"] == source) == prior_before


def test_derived_definition_with_different_signature_does_not_inherit_input_slots(saved_input_contract):
    saved, source, spec, command_spec, _, _ = saved_input_contract
    library = KnowledgeLibrary.from_dict(saved)
    derived, _, _ = derived_condition(library, source, spec, command_spec, same_signature=False)
    abstraction = abstract_measurements(derived, spec, library)
    assert not abstraction["complete"]
    assert abstraction["unresolved_commands"]
    assert not abstraction["restored_calls"]
