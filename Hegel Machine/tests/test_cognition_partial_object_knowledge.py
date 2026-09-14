"""Unresolved focal ownership does not discard measured quantities or controls."""

from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.instrument_models import fit_instrument_model, rebind_instrument_model
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.object_induction import infer_object_model
from hegel_machine.cognition.object_interventions import propose_object_probe
from hegel_machine.cognition.partial_object_knowledge import (
    partial_object_schema, propose_partial_object_claims,
)


SETTINGS = dict(objects=3, apparatus_seed=9013, name_seed=5177, wiring_seed=313)


def purchase(lab, probe, records):
    batch = lab.acquire(probe, ("fine",))
    for index, (point, transition, row) in enumerate(zip(probe.sequence, batch.transitions, batch.observations)):
        records.append({"controls": point, **transition, "target": row.target,
            "probe_id": probe.probe_id, "position": index, "history": list(probe.sequence[:index])})
    return batch.observations


@pytest.fixture(scope="module")
def partial():
    lab, records, budget = AnonymousObjectLab(readout="total", **SETTINGS), [], SearchBudget(240000)
    purchase(lab, lab.random_probe(9017, length=12), records)
    model = infer_object_model(lab.spec_for(("fine",)), records, budget)
    for serial in range(100):
        probe, _ = propose_object_probe(lab.spec_for(("fine",)), model, records, serial)
        if probe is None:
            break
        purchase(lab, probe, records)
        model = infer_object_model(lab.spec_for(("fine",)), records, budget)
    assert not model["complete"] and model["schema"] is None
    assert all({"q", "u", "x"} <= set(group) and "f" not in group for group in model["groups"])
    return model, records


@pytest.mark.parametrize("readout", ["total", "filtered_total", "event_of_total"])
def test_partial_roles_predict_totals_and_their_temporal_compositions(partial, readout):
    model, _ = partial
    saved = json.dumps(model, sort_keys=True)
    lab = AnonymousObjectLab(readout="total" if readout == "event_of_total" else readout, **SETTINGS)
    spec = replace(lab.spec_for(("fine",)), context={"object_model": model})

    def rows(seed):
        acquired = lab.acquire(lab.random_probe(seed, length=32), ("fine",)).observations
        if readout == "event_of_total":
            # This task is the observed change in total response, not the lab's
            # distinct focal-exchange readout. No hidden transition is consulted.
            return [replace(row, target=row.target - acquired[index - 1].target if index else 0.)
                    for index, row in enumerate(acquired)]
        return acquired

    training, validation = [*rows(12017), *rows(18017)], rows(24017)
    budget = SearchBudget(30000)
    proposals = propose_partial_object_claims(spec, training, budget)
    chosen = min(proposals, key=lambda item: (round(item["training_loss"], 10), item["candidate"].program.size))
    assert loss(chosen["candidate"].program, validation, 0., .5) < 1e-7
    known = {group[role] for group in model["groups"] for role in ("q", "u", "x")}
    assert all(item["candidate"].program.required_features() <= known for item in proposals)
    assert chosen["research_program"]["status"] == "requires_fresh_validation"
    assert chosen["partial_object_knowledge"]["unresolved"]
    assert json.dumps(model, sort_keys=True) == saved
    assert 0 < budget.spent <= budget.limit


def test_partial_control_knowledge_calibrates_unassigned_indicators_without_assigning_owners(partial):
    model, records = partial
    lab = AnonymousObjectLab(readout="total", **SETTINGS)
    spec = replace(lab.spec_for(("fine",)), context={"object_model": model})
    pairs = [(Observation(row["controls"], 0.), Observation(row["before"], 0.)) for row in records]
    calibration = fit_instrument_model(lab.spec_for(()), spec, pairs, SearchBudget(24000))
    restored = json.loads(json.dumps(calibration))
    binding = rebind_instrument_model(restored, spec)
    assert restored["source_schema"] is None and set(binding) == set(spec.feature_names)
    assert all("f" not in group for group in restored["source_object_model"]["groups"])
    probe = lab.random_probe(30017, length=32)
    validation = lab.acquire(probe, ("fine",)).observations
    for name, program in binding.items():
        error = sum((program.predict(Observation(point, 0.)) - row.features[name]) ** 2
                    for point, row in zip(probe.sequence, validation)) / len(validation)
        assert error < 1e-7
    indicators = set(model["focus_candidates"])
    assert indicators <= set(binding)
    assert all(restored["fields"][name]["role"] == "f" for name in indicators)
    assert all(restored["fields"][name]["members"] == [] for name in indicators)
    sparse = partial_object_schema(model)
    assert all("f" not in roles for roles in sparse["fields"].values())


def test_new_focal_evidence_completes_ownership_without_replacing_known_roles(partial):
    previous, _ = partial
    lab, records, budget = AnonymousObjectLab(readout="focus", **SETTINGS), [], SearchBudget(240000)
    purchase(lab, lab.random_probe(36017, length=12), records)
    model = infer_object_model(lab.spec_for(("fine",)), records, budget, prior=previous)
    for serial in range(100):
        probe, _ = propose_object_probe(lab.spec_for(("fine",)), model, records, serial)
        if probe is None or model["complete"]:
            break
        purchase(lab, probe, records)
        model = infer_object_model(lab.spec_for(("fine",)), records, budget, prior=previous)
    assert model["complete"]
    old = {group["q"]: group for group in previous["groups"]}
    assert all(all(group[role] == old[group["q"]][role] for role in ("u", "x"))
               and "f" in group for group in model["groups"])
    assert all("f" not in roles for roles in partial_object_schema(previous)["fields"].values())
    assert all("f" in roles for roles in partial_object_schema(model)["fields"].values())


def test_total_knowledge_does_not_claim_to_explain_focal_exchange(partial):
    model, _ = partial
    lab = AnonymousObjectLab(readout="event", **SETTINGS)
    spec = replace(lab.spec_for(("fine",)), context={"object_model": model})
    training = lab.acquire(lab.random_probe(42017, length=48), ("fine",)).observations
    validation = lab.acquire(lab.random_probe(48017, length=48), ("fine",)).observations
    proposals = propose_partial_object_claims(spec, training, SearchBudget(30000))
    chosen = min(proposals, key=lambda item: item["training_loss"])
    assert loss(chosen["candidate"].program, validation, 0., .5) > 1e-4
    assert not model["complete"]
