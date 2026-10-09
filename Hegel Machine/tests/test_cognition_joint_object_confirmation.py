"""Joint role confirmation consumes the response confirmation observations once."""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.object_induction import infer_object_model
from hegel_machine.cognition.object_induction_continuation import (
    confirm_joint_object_roles,
)
from hegel_machine.cognition.object_interventions import propose_object_probe


def _discover_batches(lab):
    """Induce from public records while retaining the paid batch boundaries."""
    channels = ("fine",)
    records, batches, model = [], [], None
    fit_budget = SearchBudget(300000)
    for serial in range(160):
        probe, question = propose_object_probe(
            lab.spec_for(channels), model, records, serial
        )
        if probe is None:
            break
        observed = lab.acquire(probe, channels)
        batches.append({
            "probe": {"probe_id": probe.probe_id, "sequence": list(probe.sequence)},
            "channels": list(channels),
            "observations": [asdict(row) for row in observed.observations],
            "transitions": deepcopy(observed.transitions),
        })
        history = []
        for point, row, transition in zip(
                probe.sequence, observed.observations, observed.transitions):
            records.append({
                "controls": dict(point),
                **deepcopy(transition),
                "target": row.target,
                "probe_id": probe.probe_id,
                "position": len(history),
                "history": deepcopy(history),
            })
            history.append(dict(point))
        model = infer_object_model(lab.spec_for(channels), records, fit_budget, prior=model)
        if model["complete"]:
            return model, batches
    pytest.fail("public dependency probes did not produce a complete model")


def _pending(model):
    pending = deepcopy(model)
    pending.update({
        "complete": False,
        "schema": None,
        "status": "pending_joint_confirmation",
        "unresolved": ["independent_role_confirmation"],
        "continuation": {
            "confirmation": {
                "status": "pending",
                "schema_before": deepcopy(model["schema"]),
            }
        },
    })
    return pending


def _agent():
    return SimpleNamespace(
        _object_schemas={},
        _object_models={},
        library=SimpleNamespace(object_models=[]),
    )


def _fresh_batch(lab, serial="joint_confirmation"):
    probe = lab.random_probe(8191, length=12, name=serial)
    observed = lab.acquire(probe, ("fine",))
    return {
        "probe": {"probe_id": probe.probe_id, "sequence": list(probe.sequence)},
        "channels": ["fine"],
        "observations": [asdict(row) for row in observed.observations],
        "transitions": deepcopy(observed.transitions),
    }


def _rows(batch):
    return [Observation(row["features"], row["target"], tuple(row["history"]))
            for row in batch["observations"]]


def test_joint_confirmation_keeps_roles_when_response_candidate_fails_independently():
    with AnonymousObjectLab(objects=3, readout="focus", apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as lab:
        model, batches = _discover_batches(lab)
        fresh = _fresh_batch(lab)
        acquisitions = len(lab.acquisitions)
        agent = _agent()
        pending = _pending(model)
        actions = []
        response_loss = loss(Program("const", value=0.0), _rows(fresh), 0.0, .5)
        assert response_loss > 1e-7

        supported = confirm_joint_object_roles(
            agent, lab, ("fine",), pending, [*batches, fresh],
            SearchBudget(300000), actions,
        )

        assert supported is True
        assert len(lab.acquisitions) == acquisitions
        assert agent._object_schemas[("fine",)] == model["schema"]
        assert pending["continuation"]["confirmation"]["supported"] is True
        # This independently rejected response candidate has no authority to
        # roll back the role result established by the shared fresh batch.
        assert pending["complete"] is True and pending["schema"] == model["schema"]


def test_joint_confirmation_rejects_a_candidate_with_swapped_membership():
    with AnonymousObjectLab(objects=3, readout="focus", apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as lab:
        model, batches = _discover_batches(lab)
        fresh = _fresh_batch(lab, "joint_wrong_schema")
        pending = _pending(model)
        groups = pending["groups"]
        groups[0]["f"], groups[1]["f"] = groups[1]["f"], groups[0]["f"]
        schema = pending["continuation"]["confirmation"]["schema_before"]
        objects = schema["objects"]
        fields = schema["fields"]
        fields[objects[0]]["f"], fields[objects[1]]["f"] = (
            fields[objects[1]]["f"], fields[objects[0]]["f"]
        )
        agent = _agent()
        acquisitions = len(lab.acquisitions)

        supported = confirm_joint_object_roles(
            agent, lab, ("fine",), pending, [*batches, fresh],
            SearchBudget(300000), [],
        )

        assert supported is False
        assert len(lab.acquisitions) == acquisitions
        assert agent._object_schemas == {}
        assert pending["complete"] is False
        assert pending["schema"] is None
        assert pending["status"] == "unresolved"
        assert pending["continuation"]["confirmation"]["status"] == "unresolved"


def test_joint_confirmation_with_insufficient_compute_keeps_schema_unresolved():
    with AnonymousObjectLab(objects=3, readout="focus", apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as lab:
        model, batches = _discover_batches(lab)
        fresh = _fresh_batch(lab, "joint_budget_shortfall")
        pending = _pending(model)
        agent = _agent()
        acquisitions = len(lab.acquisitions)

        supported = confirm_joint_object_roles(
            agent, lab, ("fine",), pending, [*batches, fresh],
            SearchBudget(1), [],
        )

        assert supported is False
        assert len(lab.acquisitions) == acquisitions
        assert agent._object_schemas == {}
        assert pending["complete"] is False
        assert pending["schema"] is None
        assert "independent_role_confirmation" in pending["unresolved"]
        assert pending["continuation"]["confirmation"]["status"] == "unresolved"
