from copy import deepcopy
import json

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.object_alternatives import enumerate_object_schemas
from hegel_machine.cognition.object_induction import infer_object_model
from hegel_machine.cognition.object_interventions import propose_object_probe
from hegel_machine.cognition.structure_lab import ObjectSchema


def _model(count=2):
    quantities = [f"quantity_{index}" for index in range(count)]
    return {"model_id": "partial_measured_model", "complete": False, "schema": None,
        "focus_control": "select", "unresolved": [f"object_{index}:x" for index in range(count)],
        "groups": [{"object_id": f"object_{index}", "q": quantities[index],
            "u": f"drive_{index}", "f": f"indicator_{index}",
            "controls": {"q": f"amount_{index}", "u": f"actuator_{index}"}}
            for index in range(count)],
        "position_candidates": {f"location_{index}": {"quantities": quantities,
            "control": f"placement_{index}"} for index in range(count)},
        "focus_candidates": {f"indicator_{index}": {"quantities": [quantities[index]]}
                             for index in range(count)}}


def test_two_ambiguous_positions_become_two_executable_alternatives_without_deciding_identity():
    model = _model()
    before = deepcopy(model)
    result = enumerate_object_schemas(model, SearchBudget(100))
    assert result["status"] == "enumerated" and result["exhaustive"]
    assert result["source_model_id"] == model["model_id"]
    assert len(result["schemas"]) == 2 and 0 < result["compute_cost"] < 100
    for schema in result["schemas"]:
        restored = ObjectSchema.from_dict(json.loads(json.dumps(schema)))
        assert len(restored.feature_names) == 8
        assert {fields["x"] for fields in restored.fields.values()} == {"location_0", "location_1"}
        for obj, fields in restored.fields.items():
            assert restored.controls[obj]["x"] == model["position_candidates"][fields["x"]]["control"]
            assert fields["f"] == f"indicator_{obj[-1]}"
    assert model == before and model["schema"] is None and not model["complete"]


def test_assignment_cap_is_explicit_and_all_factorial_alternatives_can_be_recovered():
    model = _model(4)
    capped = enumerate_object_schemas(model)
    assert len(capped["schemas"]) == 16 and not capped["exhaustive"]
    all_alternatives = enumerate_object_schemas(model, max_schemas=24)
    assert len(all_alternatives["schemas"]) == 24 and all_alternatives["exhaustive"]
    assert capped["schemas"] == all_alternatives["schemas"][:16]


def test_focus_and_position_memberships_are_both_enumerated_and_fixed_assignments_are_kept():
    model = _model()
    for group in model["groups"]:
        del group["f"]
    for item in model["focus_candidates"].values():
        item["quantities"] = [group["q"] for group in model["groups"]]
    assert len(enumerate_object_schemas(model)["schemas"]) == 4
    model["groups"][0]["x"] = "location_1"
    constrained = enumerate_object_schemas(model)
    assert len(constrained["schemas"]) == 2 and constrained["exhaustive"]
    assert all(schema["fields"]["object_0"]["x"] == "location_1" for schema in constrained["schemas"])


def test_missing_role_definitions_do_not_create_unobserved_bindings():
    model = _model()
    del model["groups"][0]["u"]
    result = enumerate_object_schemas(model)
    assert not result["schemas"] and not result["exhaustive"]
    assert result["status"] == "incomplete_roles"


def test_contradicted_bijection_is_not_repaired_by_field_order():
    model = _model()
    for candidate in model["position_candidates"].values():
        candidate["quantities"] = ["quantity_0"]
    result = enumerate_object_schemas(model)
    assert result["status"] == "inconsistent" and result["exhaustive"]
    assert result["schemas"] == []


def test_budget_truncation_cannot_be_mistaken_for_exhaustive_equivalence():
    budget = SearchBudget(3)
    result = enumerate_object_schemas(_model(), budget)
    assert result["status"] == "search_exhausted" and not result["exhaustive"]
    assert result["compute_cost"] == budget.spent == 3


def test_saved_assignment_outside_the_old_prefix_gets_an_existing_slot():
    model = _model(4)
    old = deepcopy(model)
    all_schemas = enumerate_object_schemas(model, max_schemas=24)["schemas"]
    preferred = all_schemas[-1]
    assert preferred not in enumerate_object_schemas(model)["schemas"]
    result = enumerate_object_schemas(model, SearchBudget(100), max_schemas=2,
                                      preferred_schemas=[preferred, preferred])
    assert result["schemas"] == [preferred, all_schemas[0]]
    assert not result["exhaustive"]
    assert model == old and model["schema"] is None and not model["complete"]


def test_old_role_candidates_are_checked_against_current_anchors_and_controls():
    model = _model()
    ordinary = enumerate_object_schemas(model)["schemas"]
    preferred = ordinary[-1]
    variants = []
    for section, key, value in [("fields", "q", "quantity_1"),
                                ("fields", "u", "drive_1"),
                                ("fields", "f", "indicator_1"),
                                ("controls", "q", "amount_1"),
                                ("controls", "u", "actuator_1"),
                                ("controls", "x", "placement_0")]:
        altered = deepcopy(preferred)
        altered[section]["object_0"][key] = value
        variants.append(altered)
    altered = deepcopy(preferred)
    altered["focus_control"] = "other_selector"
    variants.append(altered)
    before = deepcopy(model)
    for schema in variants:
        result = enumerate_object_schemas(model, SearchBudget(100), max_schemas=1,
                                          preferred_schemas=[schema])
        assert result["schemas"] == ordinary[:1]
    assert model == before

    # A newly observed membership rules out the formerly permitted proposal.
    model["position_candidates"]["location_1"]["quantities"] = ["quantity_1"]
    result = enumerate_object_schemas(model, preferred_schemas=[preferred])
    assert preferred not in result["schemas"] and result["exhaustive"]
    assert model["schema"] is None and not model["complete"]


def test_preferred_assignment_checks_are_charged_and_do_not_claim_exhaustion():
    model = _model(4)
    preferred = enumerate_object_schemas(model, max_schemas=24)["schemas"][-1]
    budget = SearchBudget(4)
    result = enumerate_object_schemas(model, budget, max_schemas=2,
                                      preferred_schemas=[preferred])
    assert result["schemas"] == [preferred]
    assert result["compute_cost"] == budget.spent == 4
    assert not result["exhaustive"]


def test_ordinary_object_research_passes_saved_schemas_with_original_candidate_limit(monkeypatch):
    from hegel_machine.cognition import competing_object_programs, equivalent_object_programs
    from hegel_machine.cognition.models import ResearchConfig, TaskSpec
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent

    model = _model(4)
    schemas = enumerate_object_schemas(model, max_schemas=24)["schemas"]
    fields = tuple(name for group in schemas[0]["fields"].values() for name in group.values())
    bounds = {name: (-1., 1.) for group in schemas[0]["controls"].values() for name in group.values()}
    bounds["select"] = (0., 3.)
    spec = TaskSpec("paid_roles", "saved role candidates", fields, (), (),
                    feature_bounds=bounds, context={"object_model": model})
    agent = UnifiedResearchAgent(ResearchConfig(max_candidates=4))
    source = {"schema": schemas[-1], "raw_sensor_features": list(fields),
              "raw_control_features": list(bounds), "status": "identified"}
    agent.library.object_models.extend([source, {**source, "schema": schemas[-2],
                                                 "status": "counterexample"}])
    calls = []

    def shared(*args, **kwargs):
        calls.append(("shared", kwargs))
        return []

    def competing(*args, **kwargs):
        calls.append(("competing", kwargs))
        return {"entries": [], "competition": {"status": "no_supported_program",
                    "hypotheses": [], "compute_cost": 0}}

    monkeypatch.setattr(equivalent_object_programs, "propose_equivalent_object_programs", shared)
    monkeypatch.setattr(competing_object_programs, "research_competing_object_programs", competing)
    old = deepcopy(model)
    parent = {"structure": {"schema": schemas[-3]}}
    agent._object_explanations(spec, [], SearchBudget(100), actions=[], parent=parent)
    assert [call[1]["max_candidates"] for call in calls] == [4, 4]
    assert calls[-1][1]["preferred_schemas"] == [schemas[-3], schemas[-1]]
    assert calls[-1][1]["response_parent"] is parent
    assert model == old and model["schema"] is None


def test_real_anonymous_two_object_evidence_retains_the_actual_assignment_among_alternatives():
    lab = AnonymousObjectLab(objects=2, readout="focus", apparatus_seed=9013,
                             name_seed=2917, wiring_seed=317)
    spec, records, budget = lab.spec_for(("fine",)), [], SearchBudget(120000)
    model = infer_object_model(spec, records, budget)
    for serial in range(130):
        probe, _ = propose_object_probe(spec, model, records, serial)
        if probe is None:
            break
        batch = lab.acquire(probe, ("fine",))
        for position, (point, transition, observation) in enumerate(
                zip(probe.sequence, batch.transitions, batch.observations)):
            records.append({"controls": point, **transition, "target": observation.target,
                "probe_id": probe.probe_id, "position": position,
                "history": list(probe.sequence[:position])})
        model = infer_object_model(spec, records, budget)
    result = enumerate_object_schemas(model, budget)
    assert not model["complete"] and model["schema"] is None
    assert result["exhaustive"] and len(result["schemas"]) == 2
    # Evaluation truth is consulted after discovery, never as an algorithm input.
    truth = lab._evaluation_schema()
    expected = {fields["q"]: fields for fields in truth.fields.values()}
    matches = [schema for schema in result["schemas"] if
               all(fields == expected[fields["q"]] for fields in schema["fields"].values())]
    assert len(matches) == 1


def test_public_focus_evidence_resolves_membership_when_correlated_quantity_regressions_do_not():
    lab = AnonymousObjectLab(objects=2, readout="focus", apparatus_seed=17011,
                             name_seed=64010, wiring_seed=3792)
    model, _ = _discover_current_roles(lab)
    assert all(len(item["quantities"]) == 1 for item in model["focus_candidates"].values())
    assert all(item["direct_response"]["status"] == "supported"
               for item in model["focus_candidates"].values())
    assert not model["complete"]  # Position symmetry survives the focal response.
    assert len(enumerate_object_schemas(model)["schemas"]) == 2


def _discover_current_roles(lab):
    spec, records, budget = lab.spec_for(("fine",)), [], SearchBudget(180000)
    model = infer_object_model(spec, records, budget)
    for serial in range(130):
        probe, _ = propose_object_probe(spec, model, records, serial)
        if probe is None:
            break
        batch = lab.acquire(probe, ("fine",))
        for position, (point, transition, observation) in enumerate(
                zip(probe.sequence, batch.transitions, batch.observations)):
            records.append({"controls": point, **transition, "target": observation.target,
                "probe_id": probe.probe_id, "position": position,
                "history": list(probe.sequence[:position])})
        model = infer_object_model(spec, records, budget)
    return model, records


def test_local_direct_anchor_resolves_two_positions_and_disappearing_effect_restores_ambiguity():
    settings = dict(objects=2, readout="focus", apparatus_seed=9013, name_seed=2917, wiring_seed=317)
    exchange_lab = AnonymousObjectLab(**settings, variant="exchange")
    old, exchange_rows = _discover_current_roles(exchange_lab)
    anchored_lab = AnonymousObjectLab(**settings, variant="anchored")
    anchored, anchored_rows = _discover_current_roles(anchored_lab)
    assert anchored["complete"]
    assert all(item["direct_anchor"]["status"] == "supported"
               for item in anchored["position_candidates"].values())
    revised = infer_object_model(anchored_lab.spec_for(("fine",)), anchored_rows, SearchBudget(30000), prior=old)
    assert revised["complete"] and revised["contradictions"]
    expected = {fields["q"]: fields for fields in anchored_lab._evaluation_schema().fields.values()}
    assert all({role: group[role] for role in ("q", "u", "x", "f")} == expected[group["q"]]
               for group in revised["groups"])
    restored = infer_object_model(exchange_lab.spec_for(("fine",)), exchange_rows, SearchBudget(30000), prior=anchored)
    assert not restored["complete"] and restored["schema"] is None
    assert all(len(item["quantities"]) == 2 and "direct_anchor" not in item
               for item in restored["position_candidates"].values())
    assert sum(item.get("withdrawn_membership_basis") == "local_direct_effect"
               for item in restored["contradictions"]) == 2
    assert len(enumerate_object_schemas(restored)["schemas"]) == 2
