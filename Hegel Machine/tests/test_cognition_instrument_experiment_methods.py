"""Stored instrument procedures bind roles and learn only from scored readings."""

from copy import deepcopy
from dataclasses import replace
import json
from random import Random

import pytest

from hegel_machine.cognition.instrument_experiment_methods import (
    observe_instrument_method, propose_instrument_method,
)
from hegel_machine.cognition.instrument_hypotheses import (
    assess_instrument_hypotheses, plan_instrument_disagreement,
)
from hegel_machine.cognition.instrument_models import fit_instrument_model
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, Probe, SearchBudget, TaskSpec
from hegel_machine.cognition.program_experiment_methods import ProgramExperimentMethods


class Meter:
    def __init__(self, name="u", bound=1., decay=.72):
        self.name, self.decay = name, decay
        self.spec = TaskSpec(name, "public controls", (name,), (), (),
                             feature_bounds={name: (-bound, bound)}, max_probe_length=12)
        self.measured_spec = replace(self.spec, feature_names=("meter",))

    def acquire(self, probe):
        state, rows, history = 0., [], []
        for point in probe.sequence:
            state = self.decay * state + point[self.name]
            rows.append((Observation(point, 0., tuple(history)), Observation({"meter": state}, 0.)))
            history.append(point)
        return rows

    def fit(self):
        random = Random(157)
        rows = [row for index in range(6) for row in self.acquire(Probe(str(index), tuple(
            {self.name: random.uniform(*self.spec.feature_bounds[self.name])} for _ in range(3))))]
        return fit_instrument_model(self.spec, self.measured_spec, rows, SearchBudget(24000))


def measure(book, laboratory, model, plan, budget=2000):
    pairs = laboratory.acquire(plan["probe"])
    assessment = assess_instrument_hypotheses(model, laboratory.spec, laboratory.measured_spec,
                                             pairs, SearchBudget(budget), 1e-7)
    return observe_instrument_method(book, plan, assessment, model, laboratory.spec,
        task_id=laboratory.spec.task_id, observation_rows=len(pairs), measurement_cost=3 * len(pairs))


@pytest.fixture(scope="module")
def measured_method():
    laboratory = Meter()
    unresolved = laboratory.fit()
    plan = plan_instrument_disagreement(unresolved, laboratory.spec, laboratory.measured_spec, SearchBudget(2000))
    book = ProgramExperimentMethods()
    assert measure(book, laboratory, deepcopy(unresolved), plan)["action"] == "learn_instrument_experiment_method"
    return book.to_dict(), unresolved, plan


def test_recipe_rebinds_normalized_control_roles_and_forecasts_current_law(measured_method):
    records, _, old_plan = measured_method
    book = ProgramExperimentMethods(json.loads(json.dumps(records)))
    laboratory = Meter("renamed_knob", bound=3., decay=.43)
    model = laboratory.fit()
    plan = propose_instrument_method(book, model, laboratory.spec, laboratory.measured_spec, SearchBudget(2000))
    assert plan["reason"] == "learned_instrument_experiment_method"
    assert plan["control_roles"] == ["renamed_knob"]
    assert len(plan["probe"].sequence) == len(old_plan["probe"].sequence)
    for old, current in zip(old_plan["probe"].sequence, plan["probe"].sequence):
        assert current["renamed_knob"] == pytest.approx(3 * old["u"])
    assert plan["predictions"] != old_plan["predictions"]
    assert book.to_dict() == records  # Predicting a useful experiment is not empirical support.
    assert measure(book, laboratory, model, plan)["action"] == "use_instrument_experiment_method"
    assert len(book.records[0]["support"]) == 2


def test_paid_but_partially_scored_candidates_do_not_support_or_refute_method(measured_method):
    records, unresolved, _ = measured_method
    laboratory, book, model = Meter(), ProgramExperimentMethods(records), deepcopy(unresolved)
    plan = propose_instrument_method(book, model, laboratory.spec, laboratory.measured_spec, SearchBudget(2000))
    # One candidate is actually scored, but the other receives no evaluation.
    assert measure(book, laboratory, model, plan, budget=plan["probe"].cost) is None
    assert book.to_dict() == records


def test_measured_failure_scope_blocks_the_same_laws_but_not_refitted_current_laws(measured_method):
    records, unresolved, _ = measured_method
    laboratory, book = Meter(decay=.43), ProgramExperimentMethods(records)
    plan = propose_instrument_method(book, unresolved, laboratory.spec, laboratory.measured_spec, SearchBudget(2000))
    action = measure(book, laboratory, deepcopy(unresolved), plan)
    assert action["action"] == "revise_instrument_experiment_method" and action["reason"] == "all_refuted"
    assert book.records[0]["status"] == "supported" and len(book.records[0]["support"]) == 1
    restored = ProgramExperimentMethods(json.loads(json.dumps(book.to_dict())))
    blocked = propose_instrument_method(restored, unresolved, laboratory.spec, laboratory.measured_spec, SearchBudget(2000))
    assert blocked["probe"] is None
    assert blocked["method_attempts"][0]["reason"] == "measured_failure_under_current_condition"
    refitted = laboratory.fit()
    current = propose_instrument_method(restored, refitted, laboratory.spec, laboratory.measured_spec, SearchBudget(2000))
    assert current["probe"] is not None
    assert measure(restored, laboratory, refitted, current)["action"] == "use_instrument_experiment_method"


def test_response_method_proposal_ignores_real_supported_instrument_records(measured_method):
    records, _, _ = measured_method
    book, laboratory = ProgramExperimentMethods(records), Meter()
    response_laws = [Program("feature", name="u"), Program("const", value=1.)]
    budget = SearchBudget(1000)
    assert book.propose(laboratory.spec, response_laws, budget) == []
    assert budget.spent == 0 and book.to_dict() == records


class BoundaryMeter:
    """A purchased reading can differ only at a cut or inside a narrow interval."""

    def __init__(self, strategy, *, name="u", field="meter", bounds=(-1., 1.), cuts=(.37, .37002)):
        self.name, self.field, self.cuts = name, field, cuts
        self.spec = TaskSpec(name, "public bounded control", (name,), (), (),
                             feature_bounds={name: bounds}, max_probe_length=12)
        self.measured_spec = replace(self.spec, feature_names=(field,))
        x, low, high = Program("feature", name=name), *(Program("const", value=value) for value in cuts)
        if strategy == "at":
            self.programs = [Program("ge", children=(x, low)),
                             Program("not", children=(Program("ge", children=(low, x)),))]
        else:
            self.programs = [Program("and", children=(Program("lt", children=(low, x)),
                                                      Program("lt", children=(x, high)))),
                             Program("const", value=0.)]
        self.observation_cost = 0

    def acquire(self, probe):
        rows = []
        for point in probe.sequence:
            command = Observation(point, 0.)
            rows.append((command, Observation({self.field: self.programs[0].predict(command)}, 0.)))
        self.observation_cost += len(rows)
        return rows

    def ambiguous_model(self):
        hypotheses = [{"hypothesis_id": f"{self.field}:{index}", "program": program.to_dict(),
                       "status": "training_candidate"} for index, program in enumerate(self.programs)]
        model = {"fields": {self.field: {"program": hypotheses[0]["program"], "hypotheses": hypotheses}}}
        # Both competing laws survive real paid observations away from the cut.
        prefix = Probe("prefix", tuple({self.name: value} for value in self.spec.feature_bounds[self.name]))
        assessment = assess_instrument_hypotheses(model, self.spec, self.measured_spec,
            self.acquire(prefix), SearchBudget(100), 1e-7)
        assert assessment["ambiguous_fields"] == [self.field]
        return model


@pytest.mark.parametrize("strategy", ["at", "between_adjacent"])
def test_paid_boundary_recipe_resolves_current_cuts_after_json_and_surface_change(strategy):
    source, book = BoundaryMeter(strategy), ProgramExperimentMethods()
    model = source.ambiguous_model()
    plan = plan_instrument_disagreement(model, source.spec, source.measured_spec, SearchBudget(2000))
    assert plan["sequence_mode"] == "learned_boundary"
    learned = measure(book, source, model, plan)
    assert learned is not None and learned["action"] == "learn_instrument_experiment_method"
    recipe = book.records[0]["instrument_recipe"]
    assert recipe["target_strategy"] == strategy
    assert "sequence_fractions" not in recipe
    assert "length" not in recipe

    book = ProgramExperimentMethods(json.loads(json.dumps(book.to_dict())))
    target = BoundaryMeter(strategy, name="renamed_control", field="new_readout",
                           bounds=(-4., 7.), cuts=(2.13, 2.13006))
    model = target.ambiguous_model()
    # Merely rebinding the old normalized points would leave both laws tied.
    source_low, source_high = source.spec.feature_bounds[source.name]
    target_low, target_high = target.spec.feature_bounds[target.name]
    for point in plan["probe"].sequence:
        fraction = (point[source.name] - source_low) / (source_high - source_low)
        copied = Observation({target.name: target_low + fraction * (target_high - target_low)}, 0.)
        assert len({program.predict(copied) for program in target.programs}) == 1
    previous, paid_before = deepcopy(book.to_dict()), target.observation_cost
    plan = plan_instrument_disagreement(model, target.spec, target.measured_spec,
                                       SearchBudget(2000), max_length=1, methods=book)
    assert plan["reason"] == "learned_instrument_experiment_method"
    assert plan["method_id"] == book.records[0]["method_id"]
    expected = target.cuts[0] if strategy == "at" else sum(target.cuts) / 2
    assert any(point[target.name] == pytest.approx(expected) for point in plan["probe"].sequence)
    assert plan["field_names"] == [target.field]
    assert plan["probe"].cost == 1
    assert target.observation_cost == paid_before and book.to_dict() == previous
    assert observe_instrument_method(book, plan, {}, model, target.spec,
        task_id=target.name, observation_rows=0, measurement_cost=0) is None
    assert book.to_dict() == previous
    unfunded = propose_instrument_method(book, model, target.spec, target.measured_spec, SearchBudget(0))
    assert unfunded["probe"] is None and book.to_dict() == previous
    assert measure(book, target, model, plan)["action"] == "use_instrument_experiment_method"
    assert len(book.records[0]["support"]) == 2
    assert [item["status"] for item in model["fields"][target.field]["hypotheses"]] == ["supported", "refuted"]

    # A later model with no predicted distinction is not another supported use.
    model = target.ambiguous_model()
    hypotheses = model["fields"][target.field]["hypotheses"]
    hypotheses[1]["program"] = deepcopy(hypotheses[0]["program"])
    previous, paid_before = deepcopy(book.to_dict()), target.observation_cost
    uninformative = propose_instrument_method(book, model, target.spec, target.measured_spec, SearchBudget(2000))
    assert uninformative["probe"] is None
    assert book.to_dict() == previous and target.observation_cost == paid_before
