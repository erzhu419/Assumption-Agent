"""Relation kernels can consume a paid command-to-instrument binding."""

from copy import deepcopy
import json
from dataclasses import replace

import pytest

from hegel_machine.cognition.instrument_models import instrument_binding, fit_instrument_model
from hegel_machine.cognition.instrument_hypotheses import (
    assess_instrument_hypotheses, plan_instrument_disagreement,
)
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, Probe, SearchBudget
from hegel_machine.cognition.relation_kernel_composition import collect_relation_kernels
from hegel_machine.cognition.relation_kernel_priors import rebind_relation_inputs
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from test_cognition_instrument_models import paired, resolve_purchased_hypotheses
from test_cognition_object_interventions import discover


def _relation(schema, radius):
    fields = schema["fields"]
    triples = tuple(Program("feature", name=fields[group][role])
                   for group in schema["objects"] for role in ("f", "x", "q"))
    return Program("neighbor_sum", children=(Program("const", value=radius), *triples))


def _source_prior(lab, schema):
    template = _relation(schema, .7)
    measured = lab.acquire(lab.random_probe(412, length=28), ("fine",)).observations
    paid = [replace(row, target=_relation(schema, .7).predict(row)) for row in measured]
    research = RelationIntervalResearch(template, [], SearchBudget(100000),
        structure={"fits": [{"radius": .7}], "radius_interval": [.6, .8]})
    entries = research.alternatives(paid, SearchBudget(100000))
    assert entries
    prior = deepcopy(entries[0]["research_program"])
    parameter = prior["selection_steps"][-1]["parameters"][0]
    parameter["radius_intervals"] = [[.69, .71]]
    parameter["prior_radius_intervals"] = [[.69, .71]]
    return {"source_concept_id": "paid_source_relation", "status": "supported",
            "object_schema": schema, "research_program": prior}


def test_relation_rebind_uses_paid_changed_actuator_calibration_on_command_rows():
    with UnifiedObjectLab(apparatus_seed=9013, actuator_curvature=.09) as source, \
         UnifiedObjectLab(apparatus_seed=9013, actuator_curvature=.31, rename_prefix="new_") as current:
        source_schema = json.loads(source.spec_for(("fine",)).context["object_schema"])
        current_schema = json.loads(current.spec_for(("fine",)).context["object_schema"])
        model = fit_instrument_model(current.spec, current.spec_for(("fine",)),
            paired(current, ("fine",), 183), SearchBudget(30000))
        resolve_purchased_hypotheses(current, ("fine",), model)
        calibrated = instrument_binding(model)
        assert set(calibrated) == set(current.spec_for(("fine",)).feature_names)

        # The schema supplies the source-field -> current-field identity; the
        # values are the paid command programs learned above.
        input_bindings = {}
        for group in source_schema["objects"]:
            for role, source_name in source_schema["fields"][group].items():
                current_name = current_schema["fields"][group][role]
                input_bindings[source_name] = calibrated[current_name]
        prior = _source_prior(source, source_schema)
        probe = current.random_probe(991, length=18)
        measured = current.acquire(probe, ("fine",)).observations
        history, command_rows = [], []
        for point in probe.sequence:
            command_rows.append(Observation(point, 0., tuple(history)))
            history.append(point)
        direct = _relation(current_schema, .7)
        expected = [direct.predict(row) for row in measured]

        records = collect_relation_kernels(current.spec, [prior], SearchBudget(200000),
                                           input_bindings=input_bindings)
        assert records
        rebound = next(iter(records))
        assert set(rebound.required_features()) <= set(current.spec.feature_names)
        assert [rebound.predict(row) for row in command_rows] == pytest.approx(expected, abs=1e-7)
        assert records[rebound][0]["source_concept_id"] == "paid_source_relation"
        assert records[rebound][0]["radius_intervals"] == [[.69, .71]]


def test_relation_input_binding_rejects_partial_or_unavailable_calibration():
    with UnifiedObjectLab(apparatus_seed=9013) as lab:
        schema = json.loads(lab.spec_for(("fine",)).context["object_schema"])
        prior = _source_prior(lab, schema)
        relation_features = set(_relation(schema, .7).required_features())
        partial = {name: Program("feature", name=name) for name in list(relation_features)[:-1]}
        assert collect_relation_kernels(lab.spec, [prior], SearchBudget(10000),
                                        input_bindings=partial) == {}
        unavailable = {name: Program("feature", name="missing") for name in relation_features}
        assert collect_relation_kernels(lab.spec, [prior], SearchBudget(10000),
                                        input_bindings=unavailable) == {}


def test_relation_rebinds_discovered_anonymous_roles_before_current_calibration():
    with UnifiedObjectLab(objects=3, apparatus_seed=9013, actuator_curvature=.09) as source, \
         AnonymousObjectLab(objects=3, readout="focus", apparatus_seed=9013,
                            actuator_curvature=.31, name_seed=5177, wiring_seed=313) as current:
        source_schema = json.loads(source.spec_for(("fine",)).context["object_schema"])
        prior = _source_prior(source, source_schema)
        before = deepcopy(prior)
        # Roles come from paid public interventions. The anonymous apparatus's
        # private evaluation schema is never supplied to the researcher.
        discovered, _, _ = discover(current)
        assert discovered["complete"]
        schema = discovered["schema"]
        measured_spec = replace(current.spec_for(("fine",)),
            context={"object_schema": json.dumps(schema)})

        def purchase(seed):
            probe = current.random_probe(seed, length=48)
            low, high = current.spec.feature_bounds[schema["focus_control"]]
            grid = [low + (high - low) * index / 4 for index in range(5)]
            probe = Probe(f"calibration_{seed}", tuple(
                {**point, schema["focus_control"]: grid[index]} if index < len(grid) else point
                for index, point in enumerate(probe.sequence)))
            measured = current.acquire(probe, ("fine",)).observations
            return [(Observation(point, 0., tuple(probe.sequence[:index])), row)
                    for index, (point, row) in enumerate(zip(probe.sequence, measured))]

        model = fit_instrument_model(current.spec, measured_spec,
            purchase(183), SearchBudget(30000))
        budget = SearchBudget(20000)
        assessment = assess_instrument_hypotheses(model, current.spec, measured_spec,
            purchase(719), budget, 1e-7)
        for serial in range(sum(len(field["hypotheses"]) for field in model["fields"].values())):
            if not assessment["ambiguous_fields"]:
                break
            plan = plan_instrument_disagreement(model, current.spec, measured_spec, budget, serial=serial)
            assert plan["probe"] is not None
            probe = plan["probe"]
            measured = current.acquire(probe, ("fine",)).observations
            purchased = [(Observation(point, 0., tuple(probe.sequence[:index])), row)
                         for index, (point, row) in enumerate(zip(probe.sequence, measured))]
            assessment = assess_instrument_hypotheses(model, current.spec, measured_spec,
                purchased, budget, 1e-7)
        assert not assessment["ambiguous_fields"] and not assessment["refuted_fields"]
        assert set(instrument_binding(model)) == set(measured_spec.feature_names)

        library = KnowledgeLibrary()
        library.meta_priors.append(prior)
        library.instrument_models.append(model)
        agent = UnifiedResearchAgent(library=library)
        agent._learn_instruments = True
        agent._object_schemas[("fine",)] = schema
        assert set(source.spec.feature_names).isdisjoint(current.spec.feature_names)
        assert set(source.spec_for(("fine",)).feature_names).isdisjoint(measured_spec.feature_names)
        fresh = purchase(885)
        expected = [_relation(schema, .7).predict(row) for _, row in fresh]

        for spec, rows in ((current.spec, [command for command, _ in fresh]),
                           (measured_spec, [row for _, row in fresh])):
            bindings = agent._relation_input_bindings(spec)
            records = collect_relation_kernels(spec, [prior], SearchBudget(200000),
                                               input_bindings=bindings)
            assert records
            rebound = next(iter(records))
            assert rebound.required_features() <= set(spec.feature_names)
            assert rebound.required_features().isdisjoint(source.spec.feature_names)
            assert [rebound.predict(row) for row in rows] == pytest.approx(expected, abs=1e-7)
            assert records[rebound][0]["radius_intervals"] == [[.69, .71]]
        assert prior == before


@pytest.mark.parametrize('apparatus_seed', [9013, 9017, 9029, 9041])
@pytest.mark.parametrize('curvature', [.31, .47])
def test_agent_relation_research_consumes_current_verified_instrument_binding(apparatus_seed, curvature):
    with UnifiedObjectLab(apparatus_seed=apparatus_seed, actuator_curvature=.09) as source, \
         UnifiedObjectLab(apparatus_seed=apparatus_seed, actuator_curvature=curvature, rename_prefix="new_") as current:
        source_schema = json.loads(source.spec_for(("fine",)).context["object_schema"])
        current_schema = json.loads(current.spec_for(("fine",)).context["object_schema"])
        model = fit_instrument_model(current.spec, current.spec_for(("fine",)),
            paired(current, ("fine",), 183), SearchBudget(30000))
        resolve_purchased_hypotheses(current, ("fine",), model)
        calibrated = instrument_binding(model)
        source_to_command = {}
        for group in source_schema["objects"]:
            for role, source_name in source_schema["fields"][group].items():
                source_to_command[source_name] = calibrated[current_schema["fields"][group][role]]
        source_kernel = _relation(source_schema, .7)
        rebound = rebind_relation_inputs(source_kernel, source_to_command,
                                          current.spec.feature_names)
        assert rebound is not None
        prior = _source_prior(source, source_schema)
        before = deepcopy(prior)
        probe = current.random_probe(1281, length=24)
        measured = current.acquire(probe, ('fine',)).observations
        physical_relation = _relation(current_schema, .7)
        history, rows = [], []
        for point, observed in zip(probe.sequence, measured):
            command = Observation(point, 0., tuple(history))
            rows.append(replace(command, target=physical_relation.predict(observed)))
            history.append(point)
        training, validation = rows[:16], rows[16:]
        library = KnowledgeLibrary()
        library.meta_priors.append(deepcopy(prior))
        library.instrument_models.append(model)
        library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
        agent = UnifiedResearchAgent(ResearchConfig(compute_budget=160000,
            max_question_depth=2, minimum_improvement=1e-7), library=library)
        agent._learn_instruments = True
        actions, budget = [], SearchBudget(160000)
        claims = agent._research_view(current.spec_for(()), training, validation, budget,
            actions, [], ResearchGraph(), observation_remaining=48)
        best = agent._best(claims)
        assert best is not None and best["validation_loss"] < 1e-7
        assert best["method_source_ids"] == ["paid_source_relation"]
        assert any(item["action"] == "compose_relation_kernels" for item in actions)
        assert library.meta_priors[0] == before
        library.instrument_models[0]['status'] = 'counterexample'
        assert agent._relation_input_bindings(current.spec) == {}
        assert collect_relation_kernels(current.spec, library.meta_priors, SearchBudget(10000),
            input_bindings=agent._relation_input_bindings(current.spec)) == {}
