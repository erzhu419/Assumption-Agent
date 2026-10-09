"""A relation experiment must separate the complete measured temporal response."""

import json
from dataclasses import replace

import pytest

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.competing_object_experiments import (
    assess_competing_object_probe, plan_competing_object_probe,
)
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.instrument_hypotheses import (
    assess_instrument_hypotheses, plan_instrument_disagreement,
)
from hegel_machine.cognition.instrument_models import fit_instrument_model, instrument_binding
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_structure import propose_structures
from test_cognition_object_interventions import discover


SETTINGS = dict(objects=4, readout="window_total_event", apparatus_seed=97031007,
    name_seed=97032007, wiring_seed=97033007, actuator_curvature=.6,
    relation_radius=.9687842617019095)


def _purchase(lab, probe):
    measured = lab.acquire(probe, ("fine",)).observations
    history, pairs = [], []
    for point, row in zip(probe.sequence, measured):
        pairs.append((Observation(point, row.target, tuple(history)), row))
        history.append({**point, "__target__": row.target})
    return pairs


def _calibrate(lab, spec, schema):
    def sample(seed):
        probe = lab.random_probe(seed, length=48)
        control = schema["focus_control"]
        low, high = lab.spec.feature_bounds[control]
        grid = [low + (high - low) * index / 6 for index in range(7)]
        return _purchase(lab, Probe(f"paid_calibration_{seed}", tuple(
            {**point, control: grid[index]} if index < len(grid) else point
            for index, point in enumerate(probe.sequence))))

    model = fit_instrument_model(lab.spec, spec, sample(183), SearchBudget(30000))
    budget = SearchBudget(20000)
    assessment = assess_instrument_hypotheses(model, lab.spec, spec,
        sample(719), budget, 1e-7)
    for serial in range(sum(len(field["hypotheses"]) for field in model["fields"].values())):
        if not assessment["ambiguous_fields"]:
            break
        plan = plan_instrument_disagreement(model, lab.spec, spec, budget, serial=serial)
        assert plan["probe"] is not None
        assessment = assess_instrument_hypotheses(model, lab.spec, spec,
            _purchase(lab, plan["probe"]), budget, 1e-7)
    assert not assessment["ambiguous_fields"] and not assessment["refuted_fields"]
    return instrument_binding(model)


@pytest.fixture(scope="module")
def measured_joint_response():
    with AnonymousObjectLab(**SETTINGS) as lab:
        objects, _, _ = discover(lab)
        assert objects["complete"]
        schema = objects["schema"]
        spec = replace(lab.spec_for(("fine",)), context={"object_schema": json.dumps(schema)})
        paid = [row for seed in (89, 90, 91)
                for _, row in _purchase(lab, lab.random_probe(seed, length=24))]
        entry = propose_structures(spec, paid, SearchBudget(200000))[0]
        assert entry["kind"] == "structural:joint_window_event"
        assert entry["training_loss"] < 1e-7
        binding = _calibrate(lab, spec, schema)
        assert set(binding) == set(spec.feature_names)
        yield lab, entry, paid, binding


@pytest.mark.parametrize("serial", [0, 7])
def test_paid_joint_response_probes_reduce_range_with_real_temporal_evidence(measured_joint_response, serial):
    lab, entry, original_paid, binding = measured_joint_response
    paid, experiments = list(original_paid), []
    budget = SearchBudget(40000)
    research = RelationIntervalResearch(entry["candidate"].program, [], budget,
        structure=entry["structure"])
    width = sum(high - low for item in research.parameters for low, high in item["radius_intervals"])
    for offset in range(3):
        alternatives = research.alternatives(paid, budget, experiments=experiments)
        assert len(alternatives) >= 2
        plan = plan_competing_object_probe(lab.spec, alternatives, budget,
            binding=binding, serial=serial + offset, length=5,
            questions=research.questions(alternatives))
        assert plan["forecast_status"] == "predicted_distinction"
        assert plan["probe"] is not None and plan["predicted_separation"] > 1e-7
        pairs = _purchase(lab, plan["probe"])
        evidence = assess_competing_object_probe(plan, pairs, budget)
        assert evidence["status"] == "distinguished"
        assert evidence["calibration_status"] == "supported_on_probe"
        observed = [row for _, row in pairs]
        experiments.append(observed)
        paid.extend(observed)
        revised = research.alternatives(paid, budget, experiments=experiments)
        assert revised
        new_width = sum(high - low for item in research.parameters for low, high in item["radius_intervals"])
        assert new_width < width
        assert all(loss(item["candidate"].program, observed, 0., .5) < 1e-7 for item in revised)
        research.adopt(revised[0])
        width = new_width
    assert 0 < budget.spent < budget.limit


@pytest.mark.parametrize("use_memory", [True, False])
def test_controller_researches_joint_boundaries_before_claiming_confirmation(use_memory):
    config = ResearchConfig(observation_budget=384, compute_budget=240000,
        max_rounds=6, minimum_improvement=1e-7, seed=97034007, use_memory=use_memory)
    with AnonymousObjectLab(**SETTINGS, seed=97034007, profile="confirmation") as lab:
        report = UnifiedResearchAgent(config, measurement_budget=4608).run(lab)
        assert report["status"] == "completed"
        actions = report["actions"]
        conditioned = [action for action in actions
                       if action["action"] == "condition_inherited_relation_intervals"]
        assert conditioned
        assert any(action["action"] == "test_inherited_relation_predictions"
                   and action["evidence"]["status"] == "distinguished" for action in actions)
        first_plan = next(index for index, action in enumerate(actions)
                          if action["action"] == "plan_inherited_relation_experiment")
        last_test = max(index for index, action in enumerate(actions)
                        if action["action"] == "test_inherited_relation_predictions")
        # Current measured calibration remains usable when cross-task memory
        # is disabled; otherwise every interval would request the same fit.
        assert not any(action["action"] == "research_instrument_mapping"
                       for action in actions[first_plan:last_test])
        # A bought distinguishing experiment must still have enough computation
        # to select a surviving law before the fresh confirmation is purchased.
        assert all(action["status"] != "compute_exhausted" for action in conditioned)
        widths = [sum(high - low for item in action["intervals"]
                      for low, high in item["radius_intervals"]) for action in conditioned]
        assert all(after < before for before, after in zip(widths, widths[1:]))
        fresh = lab.evaluation_rows(report["selected_channels"], seed=98034010,
                                    length=48, trajectories=2)
        assert loss(Program.from_dict(report["program"]), fresh, report["fallback"], .5) < 1e-7
