import json

import pytest

from hegel_machine.cognition.abstractions import extract_templates
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget
from hegel_machine.cognition.scale_bindings import (
    aggregate_measurement_binding, aggregate_role_binding, bind_measurement_arguments,
)
from hegel_machine.cognition.structural_discovery import discover_structure
from hegel_machine.cognition.structure_lab import ObjectSchema, StructureLab
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def schema(lab, channels):
    return json.loads(lab.spec_for(channels).context["object_schema"])


def learned_total():
    lab = UnifiedObjectLab(readout="total", apparatus_seed=211)
    batch = lab.acquire(lab.random_probe(607, length=20), ("coarse",))
    source_schema = schema(lab, ("coarse",))
    found = discover_structure(batch.observations, ObjectSchema.from_dict(source_schema),
                               budget=SearchBudget(5000), transitions=batch.transitions)
    library = KnowledgeLibrary()
    source_id, _ = library.learn(found.candidates[0], lab.spec_for(("coarse",)), batch.observations)
    return lab, library, source_id, source_schema


def bind_source(library, source_id, source_schema, current_schema):
    template = next(item for item in library.abstracts if item.template_id == f"{source_id}:abstract:0")
    definition = next(item for item in library.definitions.values()
                      if library.definition_bindings[item.definition_id]["template_id"] == template.template_id)
    arguments = library.definition_bindings[definition.definition_id]["arguments"]
    mapping = aggregate_role_binding(template, source_schema, current_schema)
    assert mapping is not None
    return Program("call", name=definition.definition_id, definition=definition,
                   children=tuple(bind_measurement_arguments(argument, mapping) for argument in arguments))


@pytest.mark.parametrize("objects,prefix", [(4, ""), (4, "sensor_"), (6, "larger_")])
def test_learned_coarse_total_calls_predict_fine_observations_after_scale_and_count_changes(objects, prefix):
    _, library, source_id, source_schema = learned_total()
    target = UnifiedObjectLab(readout="total", objects=objects, rename_prefix=prefix, apparatus_seed=211)
    call = bind_source(library, source_id, source_schema, schema(target, ("fine",)))
    rows = target.evaluation_rows(("fine",), seed=913, length=12, trajectories=2)
    assert loss(call, rows, 0., .5) < 1e-8
    assert call.definition.source_concept_id == source_id
    assert all(name.startswith("measured_" + prefix) for name in call.required_features())


def test_position_binding_weights_unequal_groups_and_additive_roles_keep_totals():
    lab = StructureLab(objects=5)
    source_schema, current_schema = lab.schema_for(scale=5), lab.schema_for(scale=2)
    fields = source_schema.fields[source_schema.objects[0]]
    program = Program("object_sum", children=tuple(Program("feature", name=name) for name in fields.values()))
    template = extract_templates(program, "whole", object_schema=source_schema.to_dict())[0]
    mapping = aggregate_role_binding(template, source_schema, current_schema)
    assert mapping is not None
    probe = lab.random_probe(1043, length=4)
    coarse, grouped = lab.observe(probe, scale=5), lab.observe(probe, scale=2)
    for expected, row in zip(coarse, grouped):
        for role, original in zip(template.roles, template.source_features):
            assert mapping[role].predict(row) == pytest.approx(expected.features[original])


def test_aggregate_lag_arguments_preserve_guarded_startup_and_frozen_source():
    lab = UnifiedObjectLab(readout="total", apparatus_seed=311)
    source_schema = schema(lab, ("coarse",))
    group = source_schema["objects"][0]
    fields = source_schema["fields"][group]
    previous = Program("add", children=(Program("lag", name=fields["q"], lag=1),
                                         Program("lag", name=fields["u"], lag=1)))
    program = Program("if", children=(Program("available", lag=1), previous, Program("const", value=0.)))
    coarse = lab.observe(lab.random_probe(709, length=9), ("coarse",))
    rows = [Observation(row.features, coarse[index - 1].target if index else 0., row.history)
            for index, row in enumerate(coarse)]
    library = KnowledgeLibrary()
    source_id, _ = library.learn(Candidate("previous", program, "history"), lab.spec_for(("coarse",)), rows)
    call = bind_source(library, source_id, source_schema, schema(lab, ("fine",)))
    fine = lab.observe(lab.random_probe(809, length=7), ("fine",))
    assert call.predict(fine[0]) == 0.
    assert call.minimum_history_depth == 0
    assert [call.predict(row) for row in fine[1:]] == pytest.approx([row.target for row in fine[:-1]])
    assert call.definition is library.definitions[call.name]
    assert len(call.children) == len(call.definition.arguments)


def test_expression_arguments_survive_rename_and_json_with_original_source_definition():
    lab, library, source_id, source_schema = learned_total()
    call = bind_source(library, source_id, source_schema, schema(lab, ("fine",)))
    rename = {name: "again_" + name for name in call.required_features()}
    renamed = bind_measurement_arguments(call, rename)
    assert renamed.definition is call.definition
    restored = Program.from_dict(json.loads(json.dumps(renamed.to_dict())))
    for row in lab.evaluation_rows(("fine",), seed=433, length=5, trajectories=1):
        changed = Observation({rename.get(name, name): value for name, value in row.features.items()}, row.target,
                              tuple({rename.get(name, name): value for name, value in past.items()}
                                    for past in row.history))
        assert restored.predict(changed) == pytest.approx(row.target)
    assert restored.definition.source_concept_id == source_id
    assert restored.definition == call.definition


def test_bindings_decline_local_source_groups_and_incomplete_target_views():
    lab = StructureLab(objects=4)
    fine, coarse = lab.schema_for(), lab.schema_for(scale=4)
    fine_template = extract_templates(Program("add", children=(Program("feature", name="q_o0"),
        Program("feature", name="u_o0"))), "local", object_schema=fine.to_dict())[0]
    assert aggregate_role_binding(fine_template, fine, coarse) is None
    _, library, source_id, source_schema = learned_total()
    template = next(item for item in library.abstracts if item.template_id == f"{source_id}:abstract:0")
    assert aggregate_role_binding(template, source_schema, lab.schema_for(objects=("o0", "o1"))) is None
    assert aggregate_role_binding(template, None, fine) is None


def test_bound_coarse_predictor_can_face_fine_instrument_residual_evidence():
    lab = UnifiedObjectLab(readout="total_event", apparatus_seed=9013, rename_prefix="later_")
    coarse_schema, fine_schema = schema(lab, ("coarse",)), schema(lab, ("fine",))
    fields = coarse_schema["fields"][coarse_schema["objects"][0]]
    program = Program("window_sum", lag=3, children=(Program("add", children=(
        Program("feature", name=fields["q"]), Program("feature", name=fields["u"]))),))
    mapping = aggregate_measurement_binding(coarse_schema, fine_schema)
    rebound = bind_measurement_arguments(program, mapping)
    probe = lab.random_probe(913, length=8)
    fine = lab.observe(probe, ("fine",))
    coarse = lab.project_rows(fine, ("fine",), ("coarse",))
    assert [rebound.predict(row) for row in fine] == pytest.approx([program.predict(row) for row in coarse])
    assert set(rebound.required_features()) <= set().union(
        *(set(item.values()) for item in fine_schema["fields"].values()))


def test_whole_apparatus_recall_keeps_its_scope_when_measurement_scale_changes():
    lab, library, source_id, source_schema = learned_total()
    library.meta_priors.append({"source_concept_id": source_id, "object_schema": source_schema})
    researcher = UnifiedResearchAgent(ResearchConfig(), library=library)
    assert len(researcher._recall(lab.spec_for(("coarse",)))) == 1
    calls = researcher._recall(lab.spec_for(("fine",)))
    assert len(calls) == 1
    required = {f"measured_{role}_o{index}" for role in ("q", "u") for index in range(4)}
    assert calls[0].required_features() == required
    for row in lab.evaluation_rows(("fine",), seed=609, length=8, trajectories=1):
        assert calls[0].predict(row) == pytest.approx(row.target)
