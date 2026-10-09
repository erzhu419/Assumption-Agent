"""Paid contrasts shrink experiments without losing coupled distinctions."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.object_dependency_methods import ObjectDependencyMethods
from hegel_machine.cognition.object_induction import infer_object_model
from hegel_machine.cognition.object_interventions import propose_object_probe
from test_cognition_object_dependency_methods import research
from test_cognition_object_interventions import acquire_records


def test_resolved_excitation_finishes_other_memberships_with_two_states():
    with AnonymousObjectLab(objects=3, readout="event", apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as lab:
        budget = SearchBudget(400000)
        records, model = [], None
        methods = ObjectDependencyMethods(budget=budget)
        residual_rows, residual_fields = 0, set()
        history_quantities, early_resolution = set(), False
        for serial in range(80):
            probe, question = propose_object_probe(lab.spec, model, records, serial, methods=methods)
            assert probe is not None
            resolved = {c["quantities"][0] for c in (model or {}).get("focus_candidates", {}).values()
                        if len(c["quantities"]) == 1}
            early_resolution |= bool(resolved) and len(history_quantities) < 3
            if question.get("quantity_candidate") in resolved:
                residual_rows += probe.cost
                residual_fields.add(question["selected_field"])
            if question.get("matched_prefix_rows") == 1:
                history_quantities.add(question["quantity_candidate"])
            acquire_records(lab, probe, records)
            if question.get("interpret_dependencies", True):
                model = infer_object_model(lab.spec, records, budget, prior=model)
            methods.observe(records)
            if model["complete"]:
                break
        assert model["complete"]
        assert early_resolution
        assert residual_rows == 4 and len(residual_fields) == 2
        truth = lab._evaluation_schema()
        for group in model["groups"]:
            actual = next(obj for obj in truth.objects if truth.fields[obj]["q"] == group["q"])
            assert group["f"] == truth.fields[actual]["f"]


def test_full_pilot_must_fit_but_already_paid_corners_are_reused():
    with AnonymousObjectLab(objects=3, readout="focus", apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as lab:
        budget = SearchBudget(400000)
        records, model = [], None
        methods = ObjectDependencyMethods(budget=budget)
        for serial in range(80):
            probe, question = propose_object_probe(lab.spec, model, records, serial, methods=methods)
            assert probe is not None
            if question.get("research_program"):
                break
            acquire_records(lab, probe, records)
            model = infer_object_model(lab.spec, records, budget, prior=model)
        cost = question["remaining_pilot_rows"]
        assert cost > probe.cost
        before = lab.observation_cost
        refused, _ = propose_object_probe(lab.spec, model, records, serial,
            methods=deepcopy(methods), remaining_rows=cost - 1)
        assert refused is None and lab.observation_cost == before
        remaining = cost
        while remaining:
            probe, question = propose_object_probe(lab.spec, model, records, serial,
                methods=methods, remaining_rows=remaining)
            assert probe is not None and question["remaining_pilot_rows"] == remaining
            acquire_records(lab, probe, records)
            remaining -= probe.cost
            serial += 1
        methods.observe(records)
        assert lab.observation_cost - before == cost
        assert methods.snapshot()[0]["status"] == "discriminating"


@pytest.mark.parametrize("objects", [3, 4])
def test_history_evidence_survives_reset_order(objects):
    with AnonymousObjectLab(objects=objects, readout="event", apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as lab:
        model, records, _, _ = research(lab)
        reversed_model = infer_object_model(lab.spec, list(reversed(records)), SearchBudget(400000))
        assert model["complete"] and reversed_model["schema"] == model["schema"]
        assert all(c["response_contexts"] for c in model["focus_candidates"].values())
