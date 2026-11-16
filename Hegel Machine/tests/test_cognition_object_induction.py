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


def _position_square(affected):
    from hegel_machine.cognition.object_induction import position_owner_predictions

    dynamic, positions = list("abcd"), list("wxyz")
    controls = {"shift": .2, "stimulus": .3, "background": 1.}
    corners = [controls, {**controls, "stimulus": 1.}, {**controls, "shift": .9},
               {**controls, "stimulus": 1., "shift": .9}]
    model = {"model_id": "measured_parent", "dynamic_fields": dynamic,
             "raw_sensor_features": dynamic + positions, "quantity_controls": {"a": "stimulus"},
             "position_candidates": {field: {"quantities": dynamic, "control": "shift" if i == 0 else field}
                                     for i, field in enumerate(positions)},
             "groups": [{"object_id": name, "q": name, "controls": {"q": name}} for name in dynamic],
             "focus_candidates": {}, "focus_control": None, "unresolved": ["focus_control"],
             "schema_alternatives": {"schemas": ["unmodified historical enumeration"]}}
    plan = {"field": "w", "excited_quantity": "a", "position_control": "shift",
            "quantity_control": "stimulus", "candidate_owners": dynamic,
            "alternative_owners": ["a", "a", "b", "d"], "corners": corners,
            "baseline_ref": {"probe_id": "p0", "position": 0},
            "owner_predictions": position_owner_predictions(dynamic, "a", dynamic)}
    records = []
    for i, point in enumerate(corners):
        before = dict(zip(dynamic + positions, [point["stimulus"], 2., 3., 4., point["shift"], -1., 0., 1.]))
        after = dict(before)
        for name in affected:
            after[name] += point["stimulus"] * point["shift"]
        records.append({"probe_id": f"p{i}", "controls": point, "before": before, "after": after,
                        "history": [], "position": 0})
    return plan, model, records


@pytest.mark.parametrize(("affected", "retained"), [
    ("", "abcd"), ("a", "abcd"), ("b", "ab"), ("bc", "a"),
])
def test_frozen_position_square_uses_observed_foreign_responses(affected, retained):
    from hegel_machine.cognition.object_induction import assess_position_experiment, apply_position_experiment
    plan, model, records = _position_square(affected)
    quote = len(records) * (len(plan["corners"][0]) + 1) + 4 * len(model["raw_sensor_features"]) + 3 * len(model["dynamic_fields"])
    work = SearchBudget(quote)
    result = assess_position_experiment(plan, model, list(reversed(records)), work)
    assert result["compute_cost"] == quote == work.spent
    assert result["retained_owners"] == list(retained)
    assert result["surviving_alternatives"] == [i for i, owner in enumerate(plan["alternative_owners"]) if owner in retained]
    updated = apply_position_experiment(model, result)
    assert updated["position_candidates"]["w"]["quantities"] == list(retained)
    if len(retained) == 4:
        assert updated is model and result["status"] == "position_domain_unchanged"
    else:
        assert result["status"] == "position_domain_narrowed" and "schema_alternatives" not in updated
        assert model["schema_alternatives"]["schemas"] and len(model["position_candidates"]["w"]["quantities"]) == 4
        assert updated["dependencies"][-1]["corner_refs"] == result["corner_refs"]
        assert apply_position_experiment(updated, result) is updated
        if len(retained) == 1:
            assert updated["groups"][0]["x"] == "w" and not any("f" in g for g in updated["groups"])


def test_position_square_retains_mismatched_or_unrealized_corners_as_unresolved():
    from hegel_machine.cognition.object_induction import assess_position_experiment, apply_position_experiment
    plan, model, records = _position_square("bd")
    later = [{**row, "position": 1, "history": [plan["corners"][0]]} for row in records]
    result = assess_position_experiment(plan, model, records[:3] + later, SearchBudget(100))
    assert result["status"] == "incomplete_position_experiment" and result["missing_corners"] == [3]
    interfered = {**records[3], "before": {**records[3]["before"], "y": 1.}}
    result = assess_position_experiment(plan, model, records[:3] + [interfered], SearchBudget(100))
    assert result["status"] == "unverified_position_intervention" and apply_position_experiment(model, result) is model


@pytest.mark.parametrize("resumed", [False, True])
@pytest.mark.parametrize("all_paid", [False, True])
def test_position_readout_reaches_parent_through_fresh_and_resumed_induction(monkeypatch, resumed, all_paid):
    from copy import deepcopy
    from types import SimpleNamespace
    from hegel_machine.cognition import object_induction, object_induction_continuation
    from hegel_machine.cognition.models import ResearchConfig, TaskSpec
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent

    plan, parent, square = _position_square("b" if all_paid else "bc")
    initial_rows = 4 if all_paid else 3
    decisions = []
    spec = TaskSpec("roles", "measure positions", tuple(parent["raw_sensor_features"]), (), (),
                    feature_bounds={key: (-1., 1.) for key in plan["corners"][0]})
    agent = UnifiedResearchAgent(ResearchConfig(use_memory=False))
    lab = SimpleNamespace(spec_for=lambda channels: spec, projectable_views=lambda source: [source],
                          project_transitions=lambda transitions, *args: transitions)
    parent.update(support_rows=initial_rows, complete=False)
    batches, actions, inferred = [], [], []
    work = SearchBudget(300)

    def append(row):
        batches.append({"probe": {"probe_id": row["probe_id"], "sequence": [row["controls"]]},
                        "channels": ["fine"], "observations": [{"target": 0., "features": row["before"], "history": []}],
                        "transitions": [{"before": row["before"], "after": row["after"]}]})

    def acquire(channels, purpose, *, rows_count=None, probe=None):
        if rows_count:
            assert not resumed
            for row in square[:initial_rows]:
                append(row)
        else:
            assert probe.sequence == (square[3]["controls"],)
            append({**square[3], "probe_id": probe.probe_id})
        return batches[-1]["observations"]

    def infer_once(spec, rows, *args, **kwargs):
        assert not resumed and len(rows) == initial_rows and not inferred
        inferred.append(len(rows))
        return deepcopy(parent)

    def observe(rows):
        methods.result = {"position_experiment": plan, "completed_relation": object_induction.assess_position_experiment(plan, parent, rows, work)}

    def propose(spec, model, rows, serial, **kwargs):
        if all_paid:
            decisions.append(model.get("position_experiment_ref"))
            if decisions == [None]:
                observe(rows)
                return None, {"position_experiment": True}
            return None, {}
        if len(rows) == 4:
            return None, {}
        return Probe("p3", (plan["corners"][3],)), {"position_experiment": True,
            "interpret_dependencies": False, "research_program": {"excitation_role": "quantity"}}

    methods = SimpleNamespace(observe=observe, snapshot=lambda: [],
        sources=[{"position_method": {"operator": "plan_position_owner_followup", "remaining_owner_count": 2}}],
        consume_position_result=lambda model: {**object_induction.apply_position_experiment(
            model, methods.result["completed_relation"]), "position_experiment_ref": {"file": "paid_decision.json"}})
    monkeypatch.setattr(agent, "_object_probe", propose)
    monkeypatch.setattr(agent, "_dependency_methods", lambda *args, **kwargs: methods)
    monkeypatch.setattr(object_induction, "infer_object_model", infer_once)
    monkeypatch.setattr(object_induction_continuation, "infer_object_model", infer_once)
    if resumed:
        for row in square[:initial_rows]:
            append(row)
    result = agent._research_objects(lab, ("fine",), acquire, batches, work, actions,
        previous_model=parent if resumed else None, remaining_rows=100)
    if all_paid:
        assert decisions == [None, {"file": "paid_decision.json"}]
        assert result["position_candidates"]["w"]["quantities"] == ["a", "b"]
        assert len(batches) == 4 and inferred == ([] if resumed else [4])
        return
    assert result["position_candidates"]["w"]["quantities"] == ["a"]
    assert result["groups"][0]["x"] == "w" and not result["complete"]
    assert parent["position_candidates"]["w"]["quantities"] == list("abcd")
    assert not next(a for a in actions if a["action"] == "test_object_dependency_question")["inference_deferred_until_paired_pilot"]
    assert len(batches) == 4 and inferred == ([] if resumed else [3])
