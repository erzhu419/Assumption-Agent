"""Paid sensor readings distinguish hypotheses that short prefixes cannot."""

from dataclasses import replace
import json
from random import Random

import pytest

from hegel_machine.cognition.instrument_hypotheses import (
    assess_instrument_hypotheses, plan_instrument_disagreement,
)
from hegel_machine.cognition.instrument_models import fit_instrument_model, instrument_binding
from hegel_machine.cognition.models import Observation, Probe, SearchBudget, TaskSpec


COMMAND = TaskSpec("unknown_meter", "bounded public input", ("u",), (), (),
                   feature_bounds={"u": (-1., 1.)}, max_probe_length=12)
MEASURED = replace(COMMAND, feature_names=("meter",))


class MemoryInstrument:
    """Two actual apparatuses with the same public interface and short readings."""

    def __init__(self, finite):
        self.finite = finite
        self.observation_cost = 0

    def acquire(self, probe):
        state, history, rows = 0., [], []
        for point in probe.sequence:
            value = point["u"]
            state = .72 * state + value
            measured = (value + sum(.72 ** lag * history[-lag]["u"]
                        for lag in range(1, min(2, len(history)) + 1))) if self.finite else state
            # Output itself gives no reason to reject a bad instrument map.
            rows.append((Observation(point, 2 * value, tuple(history)),
                         Observation({"meter": measured}, 2 * value)))
            history.append({**point, "__target__": 2 * value})
        self.observation_cost += len(rows)
        return rows


def probe(values, name="paid"):
    return Probe(name, tuple({"u": value} for value in values))


def fit_short(laboratory):
    rng = Random(157)
    rows = [row for _ in range(6)
            for row in laboratory.acquire(probe([rng.uniform(-1, 1) for _ in range(3)]))]
    return fit_instrument_model(COMMAND, MEASURED, rows, SearchBudget(24000))


@pytest.mark.parametrize("finite", [False, True])
def test_actual_longer_measurement_resolves_short_prefix_ambiguity_in_both_worlds(finite):
    laboratory = MemoryInstrument(finite)
    model = fit_short(laboratory)
    short = laboratory.acquire(probe([.4, -.2, .8]))
    assessment = assess_instrument_hypotheses(model, COMMAND, MEASURED, short, SearchBudget(1000), 1e-7)
    assert assessment["ambiguous_fields"] == ["meter"]
    assert not instrument_binding(model)
    assert len(assessment["field_losses"]["meter"]) >= 2
    assert max(assessment["field_losses"]["meter"].values()) < 1e-9

    model = json.loads(json.dumps(model))
    budget = SearchBudget(2000)
    plan = plan_instrument_disagreement(model, COMMAND, MEASURED, budget, serial=13)
    assert plan["probe"] is not None
    assert plan["probe"].cost == 4  # One genuinely unobserved step after a three-row reset.
    assert plan["field_names"] == ["meter"]
    assert plan["compute_cost"] == budget.spent > 0
    forecasts = plan["predictions"]["meter"]
    assert max(max(values) - min(values) for values in zip(*forecasts.values())) > .1
    actual = laboratory.acquire(plan["probe"])
    assessment = assess_instrument_hypotheses(model, COMMAND, MEASURED, actual, SearchBudget(1000), 1e-7)
    assert assessment["supported_fields"] == ["meter"]
    assert not assessment["ambiguous_fields"]
    survivors = [item for item in model["fields"]["meter"]["hypotheses"] if item["status"] == "supported"]
    assert len(survivors) == 1
    assert survivors[0]["kind"] == ("finite_history_polynomial" if finite else "continuous_state")
    assert model["fields"]["meter"]["program"] == survivors[0]["program"]
    assert model["fields"]["meter"]["kind"] == survivors[0]["kind"]
    assert model["observed_history_depth"] == 3

    # The surviving executable explanation predicts a fresh, longer reset.
    restored = json.loads(json.dumps(model))
    fresh = laboratory.acquire(probe([.3, -.7, .8, .2, -.5, .9, -.1, .4]))
    program = instrument_binding(restored)["meter"]
    assert max(abs(program.predict(command) - observed.features["meter"])
               for command, observed in fresh) < 1e-7
    assert laboratory.observation_cost == 18 + 3 + 4 + 8
    refuted_ids = {item["hypothesis_id"] for item in restored["fields"]["meter"]["hypotheses"]
                   if item["status"] == "refuted"}
    assess_instrument_hypotheses(restored, COMMAND, MEASURED, short, SearchBudget(1000), 1e-7)
    assert refuted_ids == {item["hypothesis_id"] for item in restored["fields"]["meter"]["hypotheses"]
                           if item["status"] == "refuted"}


def test_exhausted_budget_keeps_ambiguity_serializable_until_a_longer_probe_is_affordable():
    laboratory = MemoryInstrument(False)
    model = fit_short(laboratory)
    before = json.loads(json.dumps(model))
    paid = laboratory.acquire(probe([.4, .2, -.7]))
    assessment = assess_instrument_hypotheses(model, COMMAND, MEASURED, paid, SearchBudget(0), 1e-7)
    assert assessment["unassessed_fields"] == assessment["ambiguous_fields"] == ["meter"]
    assert assessment["compute_cost"] == 0
    assert model == before
    for budget, cap in ((SearchBudget(0), 12), (SearchBudget(1000), 0), (SearchBudget(1000), 3)):
        plan = plan_instrument_disagreement(model, COMMAND, MEASURED, budget, max_length=cap)
        assert plan["probe"] is None
        assert not plan["field_names"]
        assert model == before
    restored = json.loads(json.dumps(model))
    assert not instrument_binding(restored)
    resumed = plan_instrument_disagreement(restored, COMMAND, MEASURED, SearchBudget(2000), max_length=4)
    assert resumed["probe"].cost == 4
    assert laboratory.observation_cost == 21  # Planning itself never obtains actual readings.


def test_a_paid_field_counterexample_is_not_hidden_by_easy_rows_or_exact_response_fit():
    laboratory = MemoryInstrument(False)
    model = fit_short(laboratory)
    paid = laboratory.acquire(probe([1., 0., 0., 0.])) + laboratory.acquire(probe([0.] * 100))
    assert all(command.target == 2 * command.features["u"] == measured.target for command, measured in paid)
    assessment = assess_instrument_hypotheses(model, COMMAND, MEASURED, paid, SearchBudget(1000), .01)
    finite = next(item for item in model["fields"]["meter"]["hypotheses"]
                  if item["kind"] == "finite_history_polynomial")
    assert finite["validation_loss"] < .01 < finite["max_squared_error"]
    assert finite["status"] == "refuted"
    assert assessment["supported_fields"] == ["meter"]


def test_partially_scored_field_remains_unassessed_when_another_candidate_cannot_be_checked():
    laboratory = MemoryInstrument(False)
    model = fit_short(laboratory)
    paid = laboratory.acquire(probe([.4, -.2, .8]))
    budget = SearchBudget(len(paid))
    assessment = assess_instrument_hypotheses(model, COMMAND, MEASURED, paid, budget, 1e-7)
    assert assessment["compute_cost"] == len(paid)
    assert len(assessment["field_losses"]["meter"]) == 1
    assert assessment["unassessed_fields"] == assessment["ambiguous_fields"] == ["meter"]
    assert not assessment["supported_fields"]
    assert not instrument_binding(model)


def test_learned_membership_endpoints_are_actual_disagreement_experiments():
    from hegel_machine.cognition.instrument_experiment_methods import observe_instrument_method
    from hegel_machine.cognition.program_experiment_methods import ProgramExperimentMethods
    from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab

    with UnifiedObjectLab(apparatus_seed=9013) as laboratory:
        initial = laboratory.random_probe(183, length=48)
        batch = laboratory.acquire(initial, ("fine",))
        pairs = [(Observation(point, 0.), measured) for point, measured in zip(initial.sequence, batch.observations)]
        measured_spec = laboratory.spec_for(("fine",))
        model = fit_instrument_model(laboratory.spec, measured_spec, pairs, SearchBudget(24000))
        before = {name: len(field["hypotheses"]) for name, field in model["fields"].items()}
        plan = plan_instrument_disagreement(model, laboratory.spec, measured_spec, SearchBudget(6000), max_length=12)
        assert plan["probe"] is not None
        assert plan["sequence_mode"] == "learned_boundary"
        assert any(point["focus"] == 1.5 for point in plan["probe"].sequence)
        batch = laboratory.acquire(plan["probe"], ("fine",))
        paid = [(Observation(point, 0., tuple(plan["probe"].sequence[:index])), measured)
                for index, (point, measured) in enumerate(zip(plan["probe"].sequence, batch.observations))]
        assessment = assess_instrument_hypotheses(model, laboratory.spec, measured_spec, paid, SearchBudget(6000), 1e-7)
        assert not assessment["refuted_fields"]
        assert any(sum(item["status"] != "refuted" for item in model["fields"][name]["hypotheses"]) < before[name]
                   for name in plan["field_names"])
        book = ProgramExperimentMethods()
        feedback = observe_instrument_method(book, plan, assessment, model, laboratory.spec,
            task_id=laboratory.spec.task_id, observation_rows=len(paid), measurement_cost=batch.measurement_cost)
        assert feedback["action"] == "learn_instrument_experiment_method"
        assert book.records[0]["instrument_recipe"]["target_strategy"] == "at"
        assert "sequence_fractions" not in book.records[0]["instrument_recipe"]
