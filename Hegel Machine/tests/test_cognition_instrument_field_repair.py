"""Spend calibration on missing fields without discarding supported coordinates."""

from copy import deepcopy
from dataclasses import replace
import json
from random import Random
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.instrument_models import rebind_instrument_model
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.structure_lab import ObjectSchema
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def specifications(*, prefix="", coarse=False):
    controls = {obj: {role: f"{prefix}{obj}_{role}" for role in ("q", "u")}
                for obj in ("left", "right")}
    objects = ("both",) if coarse else ("left", "right")
    schema = ObjectSchema(objects,
        {obj: {role: f"{prefix}meter_{obj}_{role}" for role in ("q", "u")}
         for obj in objects},
        {obj: ("left", "right") if coarse else (obj,) for obj in objects},
        controls, f"{prefix}focus")
    bounds = {name: (-2., 2.) for fields in controls.values() for name in fields.values()}
    bounds[schema.focus_control] = (0., 1.)
    command = TaskSpec("local_calibration", "Two independently measured sites", tuple(bounds), (), (),
                       feature_bounds=bounds)
    return command, replace(command, feature_names=schema.feature_names,
                            context={"object_schema": json.dumps(schema.to_dict())})


def purchased(spec, seed, *, changed=False, easy=False, sequence=None):
    """The apparatus changes only left quantity gain, independently of response."""
    schema = ObjectSchema.from_dict(json.loads(spec.context["object_schema"]))
    rng = Random(seed)
    pairs = []
    points = sequence if sequence is not None else [
        {name: rng.uniform(*bounds) for name, bounds in spec.feature_bounds.items()} for _ in range(12)]
    for point in points:
        if easy:
            point[schema.controls["left"]["q"]] = 0.
        readings = {}
        for group in schema.objects:
            for role, name in schema.fields[group].items():
                readings[name] = sum(
                    (.3 + (4. if changed and member == "left" else 2.)
                     * point[schema.controls[member][role]]) if role == "q"
                    else (1. - 1.5 * point[schema.controls[member][role]])
                    for member in schema.members[group])
        pairs.append((Observation(point, 987654321.), Observation(readings, -987654321.)))
    return pairs


def learner():
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7, max_history=1),
                                 library=KnowledgeLibrary())
    agent._learn_instruments = True
    return agent


def learned(agent, command, measured):
    model = agent._fit_instruments(command, measured,
        [purchased(measured, 11), purchased(measured, 17)], SearchBudget(2400), [])
    assert set(rebind_instrument_model(model, measured)) == set(measured.feature_names)
    return model


def assert_predicts(binding, pairs):
    assert binding
    assert max(abs(program.predict(command) - actual.features[name])
               for command, actual in pairs for name, program in binding.items()) < 1e-7


def test_one_changed_field_keeps_other_calibrations_under_a_local_repair_budget():
    command, measured = specifications()
    agent = learner()
    old = learned(agent, command, measured)
    changed_name = "meter_left_q"
    retained = {name: program for name, program in agent._instrument_binding(measured, command).items()
                if name != changed_name}
    budget = SearchBudget(240)
    repaired = agent._fit_instruments(command, measured,
        [purchased(measured, 23, changed=True), purchased(measured, 31, changed=True)], budget, [],
        required_fields=(changed_name,))

    binding = agent._instrument_binding(measured, command)
    assert set(binding) == set(measured.feature_names)
    assert {name: binding[name] for name in retained} == retained
    assert_predicts(binding, purchased(measured, 43, changed=True))
    assert repaired is not None and budget.spent <= budget.limit
    assert old["status"] == "partially_supported_on_measured_trajectory"
    assert all(item["status"] == "refuted" for item in old["fields"][changed_name]["hypotheses"])


def test_partial_mapping_repairs_required_field_before_unrelated_missing_field():
    command, measured = specifications()
    agent = learner()
    old = learned(agent, command, measured)
    # A serialized partial result is a normal outcome of an earlier budget.
    old["fields"] = {name: field for name, field in old["fields"].items()
                     if name.startswith("meter_left_")}
    agent.library = KnowledgeLibrary.from_dict(json.loads(json.dumps(agent.library.to_dict())))
    pairs = [purchased(measured, 23), purchased(measured, 31)]
    first_budget = SearchBudget(200)
    first = agent._fit_instruments(command, measured, pairs, first_budget, [],
                                   required_fields=("meter_right_u",))
    binding = agent._instrument_binding(measured, command)
    assert set(binding) == {"meter_left_q", "meter_left_u", "meter_right_u"}
    assert first is not None and first_budget.spent <= first_budget.limit
    assert_predicts(binding, purchased(measured, 43))

    # The remaining right quantity still needs its schema's own control.
    second = agent._fit_instruments(command, measured,
        [*pairs, purchased(measured, 47)], SearchBudget(420), [],
        required_fields=("meter_right_q",))
    binding = agent._instrument_binding(measured, command)
    assert second is not None and set(binding) == set(measured.feature_names)
    assert_predicts(binding, purchased(measured, 53))


def test_training_fit_without_fresh_measured_support_is_not_a_substitute():
    command, measured = specifications()
    agent = learner()
    model = agent._fit_instruments(command, measured,
        [purchased(measured, 23), purchased(measured, 31, changed=True)], SearchBudget(2400), [],
        required_fields=("meter_left_q",))
    assert model is None
    assert "meter_left_q" not in agent._instrument_binding(measured, command)


def test_counterexample_survives_unfunded_repair_and_later_easy_measurements():
    command, measured = specifications()
    agent = learner()
    old = learned(agent, command, measured)
    # Four existing fields consume 4 * 24 comparisons; no fitting remains.
    agent._fit_instruments(command, measured,
        [purchased(measured, 23, changed=True), purchased(measured, 31, changed=True)],
        SearchBudget(96), [], required_fields=("meter_left_q",))
    surviving = set(measured.feature_names) - {"meter_left_q"}
    assert set(agent._instrument_binding(measured, command)) == surviving
    refuted = deepcopy(old["fields"]["meter_left_q"]["hypotheses"])

    agent._fit_instruments(command, measured,
        [purchased(measured, 43, changed=True, easy=True),
         purchased(measured, 47, changed=True, easy=True)],
        SearchBudget(0), [], required_fields=("meter_left_q",))
    assert set(agent._instrument_binding(measured, command)) == surviving
    assert old["fields"]["meter_left_q"]["hypotheses"] == refuted
    assert_predicts(agent._instrument_binding(measured, command), purchased(measured, 53, changed=True))


def test_crossed_sensor_names_rebind_counterexample_to_the_correct_source_role():
    command, measured = specifications()
    agent = learner()
    old = learned(agent, command, measured)
    new_command, new_measured = specifications(prefix="new_")
    schema = ObjectSchema.from_dict(json.loads(new_measured.context["object_schema"]))
    schema.fields["left"]["q"] = "meter_right_q"
    schema.fields["right"]["q"] = "meter_left_q"
    new_measured = replace(new_measured, feature_names=schema.feature_names,
                           context={"object_schema": json.dumps(schema.to_dict())})
    agent._fit_instruments(new_command, new_measured,
        [purchased(new_measured, 23, changed=True), purchased(new_measured, 31, changed=True)],
        SearchBudget(240), [], required_fields=("meter_right_q",))

    binding = agent._instrument_binding(new_measured, new_command)
    assert set(binding) == set(new_measured.feature_names)
    assert_predicts(binding, purchased(new_measured, 43, changed=True))
    assert all(item["status"] == "refuted" for item in old["fields"]["meter_left_q"]["hypotheses"])
    assert any(item["status"] == "supported" for item in old["fields"]["meter_right_q"]["hypotheses"])
    assert all(program.required_features() <= set(new_command.feature_names) for program in binding.values())


def test_fine_calibrations_validate_as_coarse_sums_without_inventing_inverse_fields():
    command, fine = specifications()
    agent = learner()
    old = learned(agent, command, fine)
    _, coarse = specifications(coarse=True)
    budget = SearchBudget(72)
    model = agent._fit_instruments(command, coarse,
        [purchased(coarse, 23), purchased(coarse, 31)], budget, [])
    assert model is not None
    assert set(agent._instrument_binding(coarse, command)) == set(coarse.feature_names)
    assert_predicts(agent._instrument_binding(coarse, command), purchased(coarse, 43))
    assert set(agent._instrument_binding(fine, command)) == set(fine.feature_names)
    assert old["status"] != "counterexample"
    assert budget.spent <= budget.limit

    coarse_only = learner()
    learned(coarse_only, command, coarse)
    coarse_only._fit_instruments(command, fine,
        [purchased(fine, 23), purchased(fine, 31)], SearchBudget(0), [])
    assert coarse_only._instrument_binding(fine, command) == {}


def test_refuted_coarse_projection_stays_withdrawn_without_revoking_fine_fields():
    command, fine = specifications()
    agent = learner()
    learned(agent, command, fine)
    fine_binding = agent._instrument_binding(fine, command)
    _, coarse = specifications(coarse=True)

    def shifted_coarse(seed):
        # Only the coarse quantity instrument has a changed zero point.
        return [(control, replace(actual, features={**actual.features,
                    "meter_both_q": actual.features["meter_both_q"] + 2.}))
                for control, actual in purchased(coarse, seed)]

    batches = [shifted_coarse(23), shifted_coarse(31)]
    # Pay to compare both aggregate fields, leaving no budget to fit a repair.
    agent._fit_instruments(command, coarse, batches, SearchBudget(48), [])
    assert agent._instrument_binding(fine, command) == fine_binding
    assert set(agent._instrument_binding(coarse, command)) == {"meter_both_u"}
    assert_predicts(agent._instrument_binding(fine, command), purchased(fine, 43))

    agent.library = KnowledgeLibrary.from_dict(json.loads(json.dumps(agent.library.to_dict())))
    assert set(agent._instrument_binding(coarse, command)) == {"meter_both_u"}
    repaired = agent._fit_instruments(command, coarse, batches, SearchBudget(600), [],
                                      required_fields=("meter_both_q",))
    assert repaired is not None
    assert set(agent._instrument_binding(coarse, command)) == set(coarse.feature_names)
    assert_predicts(agent._instrument_binding(coarse, command), shifted_coarse(47))
    assert agent._instrument_binding(fine, command) == fine_binding


@pytest.mark.parametrize("use_memory", [True, False])
def test_purchased_disagreement_repairs_only_refuted_field_and_keeps_supported_peers(use_memory):
    command, measured = specifications()
    agent = learner()
    old = learned(agent, command, measured)
    # Competing transferred gains are both wrong in this successor apparatus.
    field = old["fields"]["meter_left_q"]
    alternative = deepcopy(field["hypotheses"][0])
    alternative.update(hypothesis_id="opposite_gain", status="training_candidate",
        program=Program("add", children=(Program("const", value=.3),
            Program("mul", children=(Program("const", value=-2.),
                                      Program("feature", name="left_q"))))).to_dict())
    field["hypotheses"].append(alternative)
    agent.config = replace(agent.config, use_memory=use_memory)
    laboratory = SimpleNamespace(spec=command,
        spec_for=lambda channels: measured if channels else command,
        channels=[SimpleNamespace(name="meter", cost_per_row=4)])
    batches = [purchased(measured, 23, changed=True), purchased(measured, 31, changed=True)]
    acquired = []

    def acquire(channels, reason, *, probe):
        assert channels == ("meter",)
        assert reason == "distinguish_instrument_hypotheses"
        pairs = purchased(measured, 0, changed=True, sequence=probe.sequence)
        batches.append(pairs)
        acquired.append(probe)
        return [actual for _, actual in pairs]

    actions, budget = [], SearchBudget(1400)
    repaired = agent._distinguish_instruments(laboratory, old, ("meter",), acquire,
        lambda channels: batches, budget, actions, observation_allowance=4,
        required_fields=("meter_left_q",))
    assert acquired and sum(probe.cost for probe in acquired) <= 4
    assert repaired is not None
    binding = rebind_instrument_model(repaired, measured)
    assert set(binding) == set(measured.feature_names)
    assert_predicts(binding, purchased(measured, 43, changed=True))
    counterexample = next(event for event in actions
                          if event["action"] == "instrument_mapping_counterexample")
    assert set(counterexample["field_losses"]) == {"meter_left_q"}
    assert set(counterexample["retained_fields"]) == set(measured.feature_names) - {"meter_left_q"}
    repairs = [event for event in actions if event["action"] == "research_instrument_mapping"]
    assert repairs and repairs[-1]["fitted_fields"] == ["meter_left_q"]
    assert budget.spent <= budget.limit
