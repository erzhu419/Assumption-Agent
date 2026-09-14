"""Instrument predictions must generalize to actually purchased fresh readings."""

from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.instrument_models import (
    fit_instrument_model, instrument_binding, rebind_instrument_model,
)
from hegel_machine.cognition.models import Observation, Probe, SearchBudget
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab


def paired(lab, channel, seed, *, length=48):
    probe = lab.random_probe(seed, length=length)
    # Include public half-grid settings so open/closed boundary decisions have
    # measured evidence, including the apparatus's reachable exact ties.
    grid = (0., .5, 1., 1.5, 2., 2.5, 3.)
    points = tuple({**point, "focus": grid[index]} if index < len(grid) else point
                   for index, point in enumerate(probe.sequence))
    probe = Probe(f"calibrate_{seed}", points)
    batch = lab.acquire(probe, channel)
    return [(Observation(point, 987654321.), row) for point, row in zip(points, batch.observations)]


def error(binding, pairs):
    return max((program.predict(command) - measured.features[name]) ** 2
               for command, measured in pairs for name, program in binding.items())


@pytest.mark.parametrize("channel", [("fine",), ("coarse",)])
def test_purchased_data_learns_polynomial_controls_and_focus_membership(channel):
    with UnifiedObjectLab(apparatus_seed=9013) as lab:
        training = paired(lab, channel, 183)
        budget = SearchBudget(20000)
        model = fit_instrument_model(lab.spec, lab.spec_for(channel), training, budget)
        binding = instrument_binding(model)
        assert set(binding) == set(lab.spec_for(channel).feature_names), model["unmodeled_fields"]
        assert error(binding, paired(lab, channel, 885)) < 1e-10
        assert any(field.get("degree") == 3 for field in model["fields"].values())
        assert any(field["kind"] == "interval_membership" for field in model["fields"].values())
        assert 0 < budget.spent == model["compute_cost"] <= budget.limit
        assert model["status"] == "requires_fresh_validation"


def test_json_model_rebinds_renamed_measurements_and_coarsens_learned_fine_fields():
    with UnifiedObjectLab(apparatus_seed=9013) as original:
        model = fit_instrument_model(original.spec, original.spec_for(("fine",)),
            paired(original, ("fine",), 183), SearchBudget(20000))
    model = json.loads(json.dumps(model))
    with UnifiedObjectLab(apparatus_seed=9013, rename_prefix="new_") as changed:
        for channel in (("fine",), ("coarse",)):
            binding = rebind_instrument_model(model, changed.spec_for(channel))
            assert set(binding) == set(changed.spec_for(channel).feature_names)
            assert all(name.startswith("measured_new_") for name in binding)
            assert error(binding, paired(changed, channel, 885)) < 1e-10


def test_coarse_measurement_cannot_invent_individual_calibrations():
    with UnifiedObjectLab(apparatus_seed=9013) as lab:
        model = fit_instrument_model(lab.spec, lab.spec_for(("coarse",)),
            paired(lab, ("coarse",), 183), SearchBudget(20000))
        assert rebind_instrument_model(model, lab.spec_for(("fine",))) == {}


def test_public_control_role_renaming_rebinds_executable_arguments():
    with UnifiedObjectLab(apparatus_seed=9013) as lab:
        spec = lab.spec_for(("fine",))
        model = fit_instrument_model(lab.spec, spec, paired(lab, ("fine",), 183), SearchBudget(20000))
        schema = json.loads(spec.context["object_schema"])
        names = {name: f"command_{name}" for name in spec.feature_bounds}
        schema["controls"] = {member: {role: names[name] for role, name in fields.items()}
                              for member, fields in schema["controls"].items()}
        schema["focus_control"] = names[schema["focus_control"]]
        changed = replace(spec, feature_bounds={names[name]: limits for name, limits in spec.feature_bounds.items()},
                          context={**spec.context, "object_schema": json.dumps(schema)})
        binding = rebind_instrument_model(model, changed)
        fresh = [(Observation({names[name]: value for name, value in command.features.items()}, 0.), measured)
                 for command, measured in paired(lab, ("fine",), 885)]
        assert set(binding) == set(spec.feature_names)
        assert error(binding, fresh) < 1e-10


def test_changed_coefficients_require_new_measured_evidence_and_target_is_unused():
    with UnifiedObjectLab(apparatus_seed=9013) as source:
        previous = fit_instrument_model(source.spec, source.spec_for(("fine",)),
            paired(source, ("fine",), 183), SearchBudget(20000))
    with UnifiedObjectLab(apparatus_seed=451, rename_prefix="changed_") as changed:
        training = paired(changed, ("fine",), 192)
        old = rebind_instrument_model(previous, changed.spec_for(("fine",)))
        assert error(old, training) > 1e-9
        new = fit_instrument_model(changed.spec, changed.spec_for(("fine",)),
                                  training, SearchBudget(20000))
        irrelevant_targets = [(replace(command, target=-float("inf")),
                               replace(measured, target=float("nan"))) for command, measured in training]
        repeated = fit_instrument_model(changed.spec, changed.spec_for(("fine",)),
                                       irrelevant_targets, SearchBudget(20000))
        assert new == repeated
        assert error(instrument_binding(new), paired(changed, ("fine",), 884)) < 1e-10


def test_budget_exhaustion_and_unexplained_instruments_return_partial_bindings():
    with UnifiedObjectLab(apparatus_seed=9013) as lab:
        training = paired(lab, ("fine",), 183)
        budget = SearchBudget(1000)
        partial = fit_instrument_model(lab.spec, lab.spec_for(("fine",)), training, budget)
        assert 0 < len(partial["fields"]) < len(lab.spec_for(("fine",)).feature_names)
        assert partial["unmodeled_fields"]
        assert budget.spent <= 1000
        unfunded = SearchBudget(20)
        empty = fit_instrument_model(lab.spec, lab.spec_for(("fine",)), training, unfunded)
        assert instrument_binding(empty) == {}
        assert unfunded.spent == 0
        field = json.loads(lab.spec_for(("fine",)).context["object_schema"])["fields"]["o0"]["q"]
        corrupted = [(command, replace(measured, features={**measured.features,
            field: float(index % 2)})) for index, (command, measured) in enumerate(training)]
        model = fit_instrument_model(lab.spec, lab.spec_for(("fine",)), corrupted, SearchBudget(20000))
        assert field not in instrument_binding(model)
        assert field in model["unmodeled_fields"]
