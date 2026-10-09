"""Let later dependency discoveries reopen unfinished anonymous induction."""

import json
from dataclasses import replace

from .instrument_models import rebind_instrument_model
from .models import Observation
from .object_induction import validate_object_roles
from .object_induction_continuation import paid_object_records


def offer_anonymous_successor(agent, laboratory, previous, model_ids, budget):
    old = previous.get("object_research") or {}
    if previous["status"] != "unresolved" or old.get("complete") or not old.get("groups"):
        return None
    channels = tuple(laboratory.measurement_views[-1])
    raw = laboratory.spec_for(channels)
    controls = set(laboratory.spec_for(()).feature_names)
    records = paid_object_records(laboratory, previous["batches"], channels)
    if not records:
        return None

    def assignments(model):
        return {(group.get("q"), role, group[role]) for group in model.get("groups", [])
                for role in ("u", "x", "f") if group.get("q") and role in group}

    known = assignments(old)
    for model in reversed(agent.library.object_models):
        if (model.get("model_id") not in model_ids or model.get("status") == "counterexample"
                or set(model.get("raw_control_features", [])) != controls
                or set(model.get("raw_sensor_features", [])) != set(raw.feature_names)
                or not model.get("sensor_model")):
            continue
        additions = assignments(model) - known
        if not additions or len(assignments(model)) <= len(known):
            continue
        context = {**raw.context, "object_model": model}
        if model.get("schema"):
            context["object_schema"] = json.dumps(model["schema"])
        binding = rebind_instrument_model(model["sensor_model"], replace(raw, context=context))
        required = {group[role] for group in model["groups"] for role in ("q", "u", "x", "f") if role in group}
        if not required <= set(binding) or not budget.spend(len(records) * len(required)):
            continue
        errors = {name: sum((binding[name].predict(Observation(row["controls"], row["target"]))
                  - row["before"][name]) ** 2 for row in records) / len(records) for name in required}
        if max(errors.values()) > agent.config.minimum_improvement:
            continue
        role_check = validate_object_roles(model, records, budget)
        if role_check["status"] == "counterexample":
            continue
        # Calibration compatibility proposes a test; it does not establish the
        # missing memberships. The ordinary inducer must resolve those using
        # the old task's transitions and additional interventions.
        length = min(24, raw.max_probe_length)
        rate = sum(channel.cost_per_row for channel in laboratory.channels if channel.name in channels)
        return {"candidate": {"program": previous["program"],
                    "channels": previous["selected_channels"], "fallback": previous["fallback"],
                    "kind": "successor_object_model", "source_concept_ids": [],
                    "source_object_model_ids": [model["model_id"]]},
                "basis": "compatible_calibration_new_dependency_experiments",
                "old_evidence_gain": 0., "old_evidence_rows": len(records),
                "relative_gain": len(additions) / max(1, len(assignments(model))),
                "proposed_assignments": [list(item) for item in sorted(additions)],
                "calibration_losses": errors, "role_check": role_check,
                "minimum_resources": {"observation_cost": 2 * length + 12,
                    "measurement_cost": (2 * length + 12) * rate,
                    "compute_cost": 6 * len(records) + 32 * length}}
    return None
