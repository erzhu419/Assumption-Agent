"""Experiment recipes adapt to paid responses, including renamed transfers."""

import json

import pytest

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import SearchBudget, TaskSpec
from hegel_machine.cognition.object_dependency_methods import ObjectDependencyMethods
from hegel_machine.cognition.object_induction import infer_object_model
from hegel_machine.cognition.object_interventions import propose_object_probe
from test_cognition_object_interventions import acquire_records


def research(lab, sources=(), adaptive=True):
    budget = SearchBudget(400000)
    records, actions, model = [], [], None
    methods = ObjectDependencyMethods(sources, budget=budget, actions=actions) if adaptive else None
    for serial in range(160):
        probe, question = propose_object_probe(lab.spec, model, records, serial, methods=methods)
        if probe is None:
            break
        acquire_records(lab, probe, records)
        if question.get("interpret_dependencies", True):
            model = infer_object_model(lab.spec, records, budget, prior=model)
        actions.append({"action": "probe", **question})
        if methods and question.get("research_program"):
            methods.observe(records)
        if model["complete"]:
            break
    else:
        pytest.fail("method revisions kept purchasing without resolving or stopping")
    assert budget.remaining > 0
    assert lab.observation_cost == len(records)
    if methods:
        model["dependency_methods"] = methods.snapshot()
    return model, records, actions, budget.spent


@pytest.mark.parametrize("objects", [3, 4])
def test_silent_resets_select_history_and_reduce_membership_work(objects):
    params = dict(objects=objects, readout="event", apparatus_seed=9013, name_seed=5177, wiring_seed=313)
    with AnonymousObjectLab(**params) as lab:
        model, records, actions, compute = research(lab)
        truth = lab._evaluation_schema()  # Terminal check; never supplied to research.
    with AnonymousObjectLab(**params) as lab:
        old, old_records, old_actions, old_compute = research(lab, adaptive=False)
    assert model["complete"]
    for group in model["groups"]:
        actual = next(obj for obj in truth.objects if truth.fields[obj]["q"] == group["q"])
        assert all(group[role] == truth.fields[actual][role] for role in ("q", "u", "x", "f"))
        assert group["controls"] == truth.controls[actual]
    def resets(events):
        return sum(a["observation_rows"] for a in events
            if a.get("purpose") == "distinguish_membership_dependency" and a["matched_prefix_rows"] == 0)
    assert resets(actions) < resets(old_actions)
    assert compute < old_compute
    # Already purchased silent resets select history without buying another
    # reset pilot merely to rediscover that it carries no response.
    probes = [a for a in actions if a.get("research_program")]
    assert probes[0]["research_program"] == {"excitation_role": "quantity", "matched_prefix_rows": 1}
    assert resets(actions) == 0
    feedback = [a for a in actions if a["action"] == "assess_object_dependency_method"]
    assert feedback[0]["status"] == "discriminating"
    assert feedback[0]["intervention_observed"]
    assert feedback[0]["scope"]["silent_resets_with_history_signal"]
    assert any(a["status"] == "discriminating" and a["paired_probe_ids"] for a in feedback)


def test_effective_direct_method_is_kept():
    with AnonymousObjectLab(objects=3, readout="focus", apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as lab:
        model, _, actions, _ = research(lab)
    assert model["complete"]
    assert not any(a["action"] == "rewrite_object_dependency_method" for a in actions)
    assert model["dependency_methods"]
    assert all(m["matched_prefix_rows"] == 0 and m["status"] == "discriminating"
               for m in model["dependency_methods"])


def test_flat_total_response_does_not_invent_membership():
    with AnonymousObjectLab(objects=4, readout="total", apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as lab:
        model, records, _, _ = research(lab)
    assert not model["complete"] and model["schema"] is None
    assert all(len(candidate["quantities"]) == 4 for candidate in model["focus_candidates"].values())
    assert all(m["status"] == "uninformative_pilot" for m in model["dependency_methods"])
    assert len(records) < 120


def test_revised_program_survives_json_and_binds_new_task_roles():
    with AnonymousObjectLab(objects=3, readout="event", apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as lab:
        source, _, _, _ = research(lab)
    source["model_id"] = "event_source"
    library = KnowledgeLibrary()
    library.object_models.append(source)
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    params = dict(objects=4, readout="event", apparatus_seed=9917, name_seed=7103, wiring_seed=821)
    with AnonymousObjectLab(**params) as lab:
        model, records, actions, _ = research(lab, restored.object_models)
    with AnonymousObjectLab(**params) as lab:
        cold, cold_records, cold_actions, _ = research(lab)
    assert model["complete"] and model["schema"] == cold["schema"]
    assert not any(a.get("research_program") and a["matched_prefix_rows"] == 0 for a in actions)
    assert not any(a.get("research_program") and a["matched_prefix_rows"] == 0 for a in cold_actions)
    retained = [a for a in actions if a["action"] == "test_retained_dependency_method"]
    assert retained and retained[0]["source_object_model_id"] == "event_source"
    assert retained[0]["excitation_role"] == "quantity" and retained[0]["matched_prefix_rows"] == 1
    probes = [a for a in actions if a["action"] == "probe" and a.get("research_program")]
    assert probes[0]["matched_prefix_rows"] == 1
    assert not set(source["raw_sensor_features"]) & set(model["raw_sensor_features"])
    # A changed response reselects the method from its current measured scope.
    with AnonymousObjectLab(objects=4, readout="focus", apparatus_seed=9917,
                            name_seed=7103, wiring_seed=821) as lab:
        focused, _, focused_actions, _ = research(lab, restored.object_models)
    assert focused["complete"]
    assert not any(a["action"] == "test_retained_dependency_method" for a in focused_actions)
    assert all(m["matched_prefix_rows"] == 0 for m in focused["dependency_methods"])


def test_retained_method_is_revised_when_same_scope_no_longer_distinguishes():
    params = dict(objects=3, apparatus_seed=9013, name_seed=5177, wiring_seed=313)
    with AnonymousObjectLab(readout="focus", **params) as lab:
        source, _, _, _ = research(lab)
    source["model_id"] = "focus_source"
    saved = json.dumps(source, sort_keys=True)
    with AnonymousObjectLab(readout="total", **params) as lab:
        model, _, actions, _ = research(lab, [source])
    assert not model["complete"]
    assert any(a["action"] == "test_retained_dependency_method" for a in actions)
    assert any(a["action"] == "rewrite_object_dependency_method" for a in actions)
    assert all(m["status"] == "uninformative_pilot" for m in model["dependency_methods"])
    assert json.dumps(source, sort_keys=True) == saved


def test_current_measured_history_method_precedes_flat_source_after_members_resolve():
    with AnonymousObjectLab(objects=4, readout="focus", apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as lab:
        source, _, _, _ = research(lab)
    source["model_id"] = "direct_source"
    saved = json.dumps(source, sort_keys=True)
    budget, records, actions = SearchBudget(120000), [], []
    methods = ObjectDependencyMethods([source], budget=budget, actions=actions)
    with AnonymousObjectLab(objects=4, readout="window_total_event", profile="confirmation",
            apparatus_seed=97031007, name_seed=97032007, wiring_seed=97033007,
            actuator_curvature=.6, relation_radius=.9687842617019095) as lab:
        acquire_records(lab, lab.random_probe(97034007, length=12), records)
        model = infer_object_model(lab.spec, records, budget)
        for serial in range(80):
            probe, question = propose_object_probe(lab.spec, model, records, serial, methods=methods)
            assert probe is not None
            acquire_records(lab, probe, records)
            if question.get("interpret_dependencies", True):
                model = infer_object_model(lab.spec, records, budget)
            actions.append({"action": "probe", **question})
            if question.get("research_program"):
                methods.observe(records)
            if model["complete"]:
                break
        assert model["complete"]
        truth = lab._evaluation_schema()  # Only the final result is checked against hidden roles.
    first_effective = next(index for index, action in enumerate(actions)
        if action["action"] == "assess_object_dependency_method" and action["status"] == "discriminating")
    later_probes = [action["research_program"] for action in actions[first_effective + 1:]
                    if action["action"] == "probe" and action.get("research_program")]
    assert later_probes and all(recipe == {"excitation_role": "quantity", "matched_prefix_rows": 1}
                               for recipe in later_probes)
    assert any(item["status"] == "uninformative_pilot" and item["matched_prefix_rows"] == 0
               for item in methods.snapshot())
    for group in model["groups"]:
        actual = next(obj for obj in truth.objects if truth.fields[obj]["q"] == group["q"])
        assert group["f"] == truth.fields[actual]["f"]
    assert json.dumps(source, sort_keys=True) == saved


def test_current_effective_recipe_falls_back_when_remaining_members_have_flat_contrasts():
    # History exposes member a, while direct drive exposes b/c. Once a is
    # resolved, the old history success must not suppress new direct probes.
    bounds = {**{name: (0., 2.) for name in ("qa", "qb", "qc", "ua", "ub", "uc")},
              "selector": (0., 2.)}
    spec = TaskSpec("mixed_memberships", "Compare remaining memberships", tuple(bounds), (), (),
                    feature_bounds=bounds, max_probe_length=8)
    scope = {"silent_resets_with_history_signal": False}
    source = {"model_id": "direct_source", "dependency_methods": [{
        "status": "discriminating", "excitation_role": "drive", "matched_prefix_rows": 0, "scope": scope}]}
    methods = ObjectDependencyMethods([source], budget=SearchBudget(10000))
    center, records = {name: (low + high) / 2 for name, (low, high) in bounds.items()}, []

    def acquire(sequence, probe_id):
        for index, point in enumerate(sequence):
            selector = int(point["selector"])
            target = (1. + (point["qa"] if index and selector == 0 else 0.)
                      + (point["ub"] if selector == 1 else point["uc"] if selector == 2 else 0.))
            measured = {name: point[name] for name in ("qa", "qb", "qc")}
            records.append({"controls": point, "before": measured, "after": measured,
                "target": target, "history": sequence[:index], "position": index, "probe_id": probe_id})

    pairs = []
    for selector in range(3):
        baseline = {**center, "selector": float(selector)}
        changed = {**baseline, "qa": 2.}
        pair = ([baseline, baseline], [baseline, changed])
        pairs.append(pair)
        for index, sequence in enumerate(pair):
            acquire(sequence, f"history_pilot:{selector}:{index}")
    methods.observe(records)
    observed = methods._assessment(("quantity", 1), "qa", pairs, scope)
    assert observed["status"] == "discriminating"
    model = {"focus_control": "selector", "quantity_controls": {q: q for q in ("qa", "qb", "qc")},
        "drive_links": {u: {"quantity": q} for q, u in zip(("qa", "qb", "qc"), ("ua", "ub", "uc"))},
        "focus_candidates": {"fa": {"quantities": ["qa"]},
            "fb": {"quantities": ["qb", "qc"]}, "fc": {"quantities": ["qb", "qc"]}}}
    proposals = []
    for serial in range(40):
        probe, question = methods.propose(spec, model, records, serial, {"fa": 0., "fb": 1., "fc": 2.})
        assert probe is not None
        recipe = question["research_program"]
        proposals.append(recipe)
        if recipe["excitation_role"] == "drive":
            break
        acquire(list(probe.sequence), probe.probe_id)
        methods.observe(records)
    assert proposals[0] == {"excitation_role": "quantity", "matched_prefix_rows": 1}
    assert proposals[-1] == {"excitation_role": "drive", "matched_prefix_rows": 0}
    assert observed in methods.snapshot()
    assert any(item["status"] == "uninformative_pilot" and item["quantity_candidate"] in ("qb", "qc")
               for item in methods.snapshot())
