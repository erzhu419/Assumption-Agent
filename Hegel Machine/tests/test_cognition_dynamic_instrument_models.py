"""Dynamic instrument laws use purchased control histories and fresh readings."""

from dataclasses import replace
import json
from random import Random

from hegel_machine.cognition.instrument_models import fit_instrument_model, instrument_binding
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec


def field(name):
    return Program("feature", name=name)


def const(value):
    return Program("const", value=value)


def binary(op, left, right):
    return Program(op, children=(left, right))


def lag(name, depth):
    return Program("if", children=(Program("available", lag=depth),
                                   Program("lag", name=name, lag=depth), const(0.)))


def specs(names=("u",)):
    commands = TaskSpec("dynamic", "public controls", names, (), (),
                        feature_bounds={name: (-1., 1.) for name in names})
    return commands, replace(commands, feature_names=("sensor",))


def paired(program, seed, *, names=("u",), lengths=(12, 12, 12, 12, 12, 12)):
    rng, result = Random(seed), []
    for length in lengths:
        history = []
        for index in range(length):
            # Repeated current commands deliberately leave different pasts.
            features = {name: rng.choice((-.9, -.4, 0., .3, .8)) for name in names}
            commands = Observation(features, 31337. + index, tuple(history))
            reading = program.predict(commands)
            measured = Observation({"sensor": reading}, -9127.,
                                   tuple({"sensor": 8877., "__target__": -6622.} for _ in history))
            result.append((commands, measured))
            history.append({**features, "__target__": 7857. + index, "unobserved_response": 9321.})
    return result


def max_error(program, pairs):
    return max((program.predict(commands) - measured.features["sensor"]) ** 2 for commands, measured in pairs)


def test_same_current_control_different_pasts_identifies_delay_and_generalizes():
    commands, measured = specs()
    truth = binary("add", const(.2), binary("mul", const(1.7), lag("u", 2)))
    training = paired(truth, 173)
    assert any(left.features == right.features and left_read.features != right_read.features
               for left, left_read in training for right, right_read in training)
    static = fit_instrument_model(commands, measured, training, SearchBudget(24000), max_history=0)
    assert not static["fields"] and "sensor" in static["unmodeled_fields"]
    budget = SearchBudget(24000)
    model = fit_instrument_model(commands, measured, training, budget, max_history=3)
    law = instrument_binding(model)["sensor"]
    assert model["fields"]["sensor"]["kind"] == "finite_history_polynomial"
    assert model["fields"]["sensor"]["history_depth"] == 2
    assert law.history_depth == 2 and law.minimum_history_depth == 0
    assert max_error(law, paired(truth, 271, lengths=(1, 2, 3, 17, 29))) < 1e-10
    assert model["compute_cost"] == budget.spent <= budget.limit
    assert model["status"] == "requires_fresh_validation"


def test_dynamic_mixed_polynomial_uses_current_and_finite_historical_controls():
    commands, measured = specs(("u", "v"))
    truth = binary("add", binary("mul", field("u"), lag("v", 1)),
        binary("mul", const(.5), binary("mul", lag("u", 2), lag("u", 2))))
    training = paired(truth, 181, names=("u", "v"), lengths=(20,) * 6)
    model = fit_instrument_model(commands, measured, training, SearchBudget(24000), max_history=3)
    law = instrument_binding(model)["sensor"]
    assert model["fields"]["sensor"]["degree"] == 2
    assert model["fields"]["sensor"]["history_depth"] == 2
    assert {("v", 1), ("u", 2)} <= law.observed_variables()
    assert max_error(law, paired(truth, 293, names=("u", "v"), lengths=(1, 2, 17, 25))) < 1e-10


def test_each_purchased_trajectory_restarts_its_instrument_history():
    commands, measured = specs()
    truth = binary("add", binary("mul", const(.4), field("u")),
        binary("mul", const(-.8), lag("u", 1)))
    training = paired(truth, 191, lengths=(9,) * 8)
    model = fit_instrument_model(commands, measured, training, SearchBudget(24000))
    law = instrument_binding(model)["sensor"]
    fresh = paired(truth, 307, lengths=(1, 1, 2, 3, 13))
    assert max_error(law, fresh) < 1e-10
    # All reset rows use only their own current control, even following another
    # probe's nonzero final command in the flattened purchase list.
    for command, reading in fresh:
        if not command.history:
            assert abs(law.predict(command) - .4 * command.features["u"]) < 1e-8


def test_response_targets_and_measured_history_cannot_become_calibration_coordinates():
    commands, measured = specs()
    truth = binary("add", field("u"), lag("u", 1))
    training = paired(truth, 197)
    changed = [(replace(command, target=float("nan"), history=tuple(
                    {**past, "__target__": -float("inf"), "unobserved_response": float("nan")}
                    for past in command.history)),
                replace(reading, target=float("inf"), history=tuple(
                    {"sensor": -float("inf"), "__target__": float("nan")} for _ in reading.history)))
               for command, reading in training]
    original = fit_instrument_model(commands, measured, training, SearchBudget(24000))
    independent = fit_instrument_model(commands, measured, changed, SearchBudget(24000))
    assert original == independent
    law = instrument_binding(json.loads(json.dumps(independent)))["sensor"]
    assert {name for name, _ in law.observed_variables()} == {"u"}


def test_static_early_stop_keeps_its_model_and_cost_even_with_available_history():
    commands, measured = specs()
    truth = binary("add", const(.7), binary("mul", field("u"), field("u")))
    training = paired(truth, 199)
    static_budget, dynamic_budget = SearchBudget(24000), SearchBudget(24000)
    static = fit_instrument_model(commands, measured, training, static_budget, max_history=0)
    current = fit_instrument_model(commands, measured, training, dynamic_budget, max_history=4)
    assert current == static and dynamic_budget.spent == static_budget.spent
    assert current["fields"]["sensor"]["kind"] == "polynomial"
    assert instrument_binding(current)["sensor"].history_depth == 0


def test_history_limit_and_compute_budget_leave_unexplained_fields_out_of_binding():
    commands, measured = specs()
    truth = lag("u", 3)
    training = paired(truth, 211, lengths=(18,) * 8)
    too_short = fit_instrument_model(commands, measured, training, SearchBudget(24000), max_history=1)
    assert not too_short["fields"] and "sensor" in too_short["unmodeled_fields"]
    budget = SearchBudget(1000)
    unfunded = fit_instrument_model(commands, measured, training, budget, max_history=4)
    assert not unfunded["fields"] and budget.spent <= budget.limit
    sufficient = fit_instrument_model(commands, measured, training, SearchBudget(24000), max_history=3)
    assert max_error(instrument_binding(sufficient)["sensor"], paired(truth, 313, lengths=(2, 3, 21))) < 1e-10


def test_one_training_reset_does_not_create_a_separate_unmeasured_startup_law():
    # The runtime's actual 64-row trajectory contains only one reset. Adding
    # availability and its products as free coordinates formerly fitted that
    # one point but missed the next trajectory's first reading by .2773.
    from test_cognition_dynamic_instrument_runtime import COMMAND, MEASURED, pairs

    training = pairs(731)
    model = fit_instrument_model(COMMAND, MEASURED, training, SearchBudget(24000))
    law = instrument_binding(model)["meter"]
    assert model["fields"]["meter"]["degree"] == 2
    assert model["fields"]["meter"]["history_depth"] == 1
    for seed in (953, 1941, 2573):
        validation = pairs(seed)
        assert max((law.predict(command) - reading.features["meter"]) ** 2
                   for command, reading in validation) < 1e-10
        command, reading = validation[0]
        assert not command.history
        assert abs(law.predict(command) - .4 * command.features["u"]) < 1e-8
