from dataclasses import replace
import json
from random import Random

import pytest

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.models import Probe, SearchBudget
from hegel_machine.cognition.object_induction import infer_object_model, validate_object_roles
from hegel_machine.cognition.object_interventions import propose_object_probe


def _purchase(lab, probe, records):
    batch = lab.acquire(probe, ("fine",))
    for position, (point, transition, observation) in enumerate(
            zip(probe.sequence, batch.transitions, batch.observations)):
        records.append({"controls": point, **transition, "target": observation.target,
            "probe_id": probe.probe_id, "position": position,
            "history": list(probe.sequence[:position])})


def _discover(lab, *, random_prefix=False):
    spec, records, budget = lab.spec_for(("fine",)), [], SearchBudget(280000)
    if random_prefix:
        _purchase(lab, lab.random_probe(17017, length=12), records)
    model = infer_object_model(spec, records, budget)
    for serial in range(130):
        probe, _ = propose_object_probe(spec, model, records, serial)
        if probe is None or model["complete"]:
            break
        _purchase(lab, probe, records)
        model = infer_object_model(spec, records, budget)
    assert len(records) <= 170
    return model, records, budget


def _assert_actual_groups(lab, model):
    # Truth is used only by this evaluator, after public-evidence induction.
    schema = lab._evaluation_schema()
    expected = {fields["q"]: (fields, schema.controls[obj])
                for obj, fields in schema.fields.items()}
    assert model["complete"]
    assert model["schema"]["focus_control"] == schema.focus_control
    assert len(model["groups"]) == len(expected)
    for group in model["groups"]:
        fields, controls = expected[group["q"]]
        assert {role: group[role] for role in ("q", "u", "x", "f")} == fields
        assert group["controls"] == controls


@pytest.mark.parametrize("objects,variant,name_seed,wiring_seed", [
    (3, "exchange", 5177, 313),
    (4, "exchange", 5177, 313),
    (4, "sink", 1207, 719),
    (3, "nonlocal", 9919, 837),
])
def test_actual_interventions_recover_anonymous_object_groups(objects, variant, name_seed, wiring_seed):
    lab = AnonymousObjectLab(objects=objects, readout="focus", variant=variant,
        apparatus_seed=9013, actuator_curvature=.31, name_seed=name_seed, wiring_seed=wiring_seed)
    model, records, budget = _discover(lab, random_prefix=True)
    _assert_actual_groups(lab, model)
    assert model["dependencies"]
    assert any(item["distinguishing_contrasts"] for item in model["focus_candidates"].values())
    assert budget.spent < 180000  # Leave a useful budget for response learning.
    _purchase(lab, lab.random_probe(8317, length=12), records)
    confirmed = infer_object_model(lab.spec_for(("fine",)), records, SearchBudget(30000), prior=model)
    _assert_actual_groups(lab, confirmed)
    assert not confirmed["contradictions"]
    assert json.loads(json.dumps(confirmed))["schema"] == confirmed["schema"]


def test_total_response_cannot_assign_any_selector_to_an_object():
    lab = AnonymousObjectLab(objects=4, readout="total", apparatus_seed=9013,
                             name_seed=5177, wiring_seed=313)
    model, _, _ = _discover(lab)
    assert not model["complete"] and model["schema"] is None
    assert len(model["focus_candidates"]) == 4
    assert all(len(item["quantities"]) == 4 and not item["distinguishing_contrasts"]
               for item in model["focus_candidates"].values())
    assert all("f" not in group and {"q", "u", "x"} <= set(group) for group in model["groups"])


def test_two_locations_retain_observational_symmetry():
    lab = AnonymousObjectLab(objects=2, readout="focus", apparatus_seed=9013,
                             name_seed=2917, wiring_seed=317)
    model, _, _ = _discover(lab)
    assert model["schema"] is None
    assert len(model["position_candidates"]) == 2
    assert all(len(item["quantities"]) == 2 for item in model["position_candidates"].values())


@pytest.mark.parametrize("objects,name_seed,wiring_seed", [(3, 731, 97), (4, 63928, 3780)])
def test_different_quantity_backgrounds_distinguish_direct_position_effect_from_drive(objects, name_seed, wiring_seed):
    lab = AnonymousObjectLab(objects=objects, readout="focus", variant="anchored",
                             apparatus_seed=9013, name_seed=name_seed, wiring_seed=wiring_seed)
    model, _, _ = _discover(lab)
    _assert_actual_groups(lab, model)
    for group in model["groups"]:
        assert model["role_candidates"][group["x"]]["dependency_kind"] == "relational_with_direct_effect"
        assert model["role_candidates"][group["u"]]["dependency_kind"] == "pure_additive"
        assert any(hypothesis["field"] == group["x"] and hypothesis["kind"] == "pure_additive"
                   and hypothesis["status"] == "refuted" and hypothesis["counterexamples"]
                   for hypothesis in model["role_hypotheses"])


def test_changed_wiring_rebuilds_membership_and_reports_contradicted_prior():
    settings = dict(objects=3, readout="focus", apparatus_seed=9013, name_seed=9919)
    first = AnonymousObjectLab(**settings, wiring_seed=837)
    changed = AnonymousObjectLab(**settings, wiring_seed=271)
    assert first.spec_for(("fine",)) == changed.spec_for(("fine",))
    old, _, _ = _discover(first)
    new, records, _ = _discover(changed)
    restored = infer_object_model(changed.spec_for(("fine",)), records, SearchBudget(30000), prior=old)
    _assert_actual_groups(changed, new)
    _assert_actual_groups(changed, restored)
    assert old["schema"] != restored["schema"]
    assert restored["contradictions"]


def test_identification_is_independent_of_record_and_control_insertion_order():
    lab = AnonymousObjectLab(objects=3, readout="focus", apparatus_seed=9013,
                             name_seed=5177, wiring_seed=313)
    expected, records, _ = _discover(lab)
    Random(2017).shuffle(records)
    spec = lab.spec_for(("fine",))
    reordered = replace(spec, feature_names=tuple(reversed(spec.feature_names)),
                        feature_bounds=dict(reversed(list(spec.feature_bounds.items()))))
    actual = infer_object_model(reordered, records, SearchBudget(30000))
    assert actual["schema"] == expected["schema"]


def test_unmatched_trajectory_prefixes_do_not_supply_membership_response_evidence():
    lab = AnonymousObjectLab(objects=3, readout="focus", apparatus_seed=9013)
    spec = lab.spec_for(("fine",))
    center = {name: (low + high) / 2 for name, (low, high) in spec.feature_bounds.items()}
    points = [center] + [{**center, control: value}
                        for control, interval in spec.feature_bounds.items() for value in interval]
    records = []
    _purchase(lab, Probe("one_trajectory", tuple(points)), records)
    model = infer_object_model(spec, records, SearchBudget(30000))
    assert model["focus_candidates"]
    assert all(not item["distinguishing_contrasts"] for item in model["focus_candidates"].values())
    assert all(pair["target_delta"] is None
               for effect in model["control_effects"].values() for pair in effect["pairs"])
    assert not model["complete"]


def test_insufficient_compute_does_not_publish_unsupported_roles():
    lab = AnonymousObjectLab(objects=3, readout="focus", apparatus_seed=9013)
    records = []
    _purchase(lab, lab.random_probe(719, length=4), records)
    budget = SearchBudget(3)
    model = infer_object_model(lab.spec_for(("fine",)), records, budget)
    assert model["schema"] is None and not model["complete"]
    assert model["groups"] == []
    assert budget.spent <= budget.limit


def _witness_points(model):
    points = []
    for witness in model["dependency_witnesses"]:
        for point in witness["sequence"]:
            if point not in points:
                points.append(point)
    return points


def test_role_definitions_remain_competing_until_the_quantity_background_is_changed():
    lab = AnonymousObjectLab(objects=3, readout="focus", variant="anchored",
                             apparatus_seed=9013, name_seed=63928, wiring_seed=3780)
    records, model, budget = [], None, SearchBudget(100000)
    for serial in range(10):
        probe, action = propose_object_probe(lab.spec, model, records, serial)
        if action["purpose"] != "isolate_control_effects":
            break
        _purchase(lab, probe, records)
        model = infer_object_model(lab.spec, records, budget)
    assert action["purpose"] == "distinguish_role_definition"
    assert not model["complete"]
    assert len(model["role_candidates"]) == 6
    assert all(candidate["roles"] == ["u", "x"] and candidate["dependency_kind"] is None
               for candidate in model["role_candidates"].values())
    _purchase(lab, probe, records)
    revised = infer_object_model(lab.spec, records, budget, prior=model)
    supported = [item for item in revised["role_hypotheses"] if item["status"] == "supported"]
    assert sum(item["kind"] == "pure_additive" for item in supported) == 3
    assert sum(item["kind"] == "relational_with_direct_effect" for item in supported) == 3
    assert all(len(witness["sequence"]) == 4 for witness in revised["dependency_witnesses"])


@pytest.mark.parametrize("old_variant,new_variant", [("exchange", "anchored"), ("anchored", "exchange")])
def test_role_witnesses_reopen_changed_dependencies_even_when_before_calibration_is_identical(old_variant, new_variant):
    settings = dict(objects=3, apparatus_seed=9013, name_seed=63928, wiring_seed=3780)
    previous_lab = AnonymousObjectLab(**settings, readout="focus", variant=old_variant)
    current_lab = AnonymousObjectLab(**settings, readout="total", variant=new_variant)
    model, _, _ = _discover(previous_lab)
    points = _witness_points(model)
    assert len(points) <= 14
    previous, current = [], []
    _purchase(previous_lab, Probe("old_witness", tuple(points)), previous)
    _purchase(current_lab, Probe("new_witness", tuple(points)), current)
    assert [row["before"] for row in current] == [row["before"] for row in previous]
    frozen = json.loads(json.dumps(model))
    result = validate_object_roles(model, current, SearchBudget(10000))
    assert result["status"] == "counterexample"
    assert len(result["counterexamples"]) == 3
    assert all(item["records"] for item in result["counterexamples"])
    assert model == frozen
    rebuilt, _, _ = _discover(AnonymousObjectLab(**settings, readout="focus", variant=new_variant))
    _assert_actual_groups(current_lab, rebuilt)


@pytest.mark.parametrize("variant,readout", [("exchange", "total"), ("sink", "focus"), ("nonlocal", "event")])
def test_transition_witnesses_preserve_roles_when_only_response_or_nonrelational_coefficients_change(variant, readout):
    settings = dict(objects=4, apparatus_seed=9013, name_seed=317, wiring_seed=917)
    model, _, _ = _discover(AnonymousObjectLab(**settings, readout="focus", variant="exchange"))
    current = AnonymousObjectLab(**settings, readout=readout, variant=variant)
    points = _witness_points(model)
    assert len(points) <= 18
    records = []
    _purchase(current, Probe("retained_roles", tuple(points)), records)
    result = validate_object_roles(model, records, SearchBudget(10000))
    assert result["status"] == "supported" and not result["counterexamples"]
    assert len(result["tested_hypotheses"]) == 8
    assert result["compute_cost"] < 3000
    assert validate_object_roles(model, records, SearchBudget(2))["status"] == "insufficient"
