"""Measured mechanisms survive dynamic calibration replacement and recovery."""

from dataclasses import replace
import json
from random import Random

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


COMMAND = TaskSpec("dynamic", "Predict the response.", ("u",), (), (),
                   feature_bounds={"u": (-1., 1.)}, max_probe_length=96)
MEASURED = replace(COMMAND, feature_names=("u", "meter"))
CONFIG = ResearchConfig(max_history=4, minimum_improvement=1e-7)
METER = Program("feature", name="meter")
RESPONSE = Program("add", children=(Program("mul", children=(METER, METER)),
    Program("if", children=(Program("available", lag=1),
        Program("lag", name="meter", lag=1), Program("const", value=0.)))))


def pairs(seed, *, delay=1, gain=.9, response_gain=1.):
    rng, history, command_history, result = Random(seed), [], [], []
    for _ in range(64):
        value = rng.uniform(-1., 1.)
        previous = command_history[-delay]["u"] if len(command_history) >= delay else 0.
        reading = .4 * value + gain * previous + .2 * previous ** 2
        row = Observation({"u": value, "meter": reading}, 0., tuple(history))
        target = response_gain * RESPONSE.predict(row)
        row = replace(row, target=target)
        result.append((Observation({"u": value}, target, tuple(command_history)), row))
        history.append({**row.features, "__target__": target})
        command_history.append({"u": value, "__target__": target})
    return result


def learned():
    library = KnowledgeLibrary()
    agent = UnifiedResearchAgent(CONFIG, library=library)
    training, validation = pairs(731), pairs(953)
    model = agent._fit_instruments(COMMAND, MEASURED, [training, validation], SearchBudget(90000), [])
    assert model and model["fields"]["meter"]["history_depth"] >= 1
    source, _ = library.learn(Candidate("measured_response", RESPONSE, "research"),
                              MEASURED, [row for _, row in training])
    library.meta_priors.append({"source_concept_id": source, "source_features": list(MEASURED.feature_names),
                               "control_features": list(COMMAND.feature_names), "object_schema": None})
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    agent = UnifiedResearchAgent(CONFIG, library=restored)
    agent._learn_instruments = agent._revise_inputs = True
    return agent, source


def test_restored_measured_call_rebinds_dynamic_instruments_including_lagged_readings():
    agent, source = learned()
    calls = agent._recall(COMMAND)
    assert calls and all(call.definition.source_concept_id == source for call in calls)
    fresh = pairs(1941)
    for call in calls:
        assert {name for name, _ in call.observed_variables()} <= {"u"}
        assert max(abs(call.predict(command) - row.target) for command, row in fresh) < 1e-7
        assert call.minimum_history_depth == 0


def test_purchased_lag_drift_replaces_calibration_without_relearning_measured_mechanism():
    agent, source = learned()
    changed = pairs(1941, delay=2, gain=-.65)
    old_calls = agent._recall(COMMAND)
    assert max(abs(old_calls[0].predict(command) - row.target) for command, row in changed) > .1
    actions = []
    model = agent._fit_instruments(COMMAND, MEASURED,
        [changed, pairs(2573, delay=2, gain=-.65)], SearchBudget(90000), actions)
    assert model and model["fields"]["meter"]["history_depth"] >= 2
    assert any(action["action"] == "instrument_mapping_counterexample" for action in actions)
    assert any(item["status"] == "counterexample" for item in agent.library.instrument_models)
    for call in agent._recall(COMMAND):
        assert call.definition.source_concept_id == source
        assert max(abs(call.predict(command) - row.target)
                   for command, row in pairs(3319, delay=2, gain=-.65)) < 1e-7
    assert len(agent.library.concepts) == 1


def test_response_drift_keeps_a_field_model_supported_by_actual_readings():
    agent, _ = learned()
    previous = agent.library.instrument_models[-1]
    actions = []
    model = agent._fit_instruments(COMMAND, MEASURED,
        [pairs(1941, response_gain=2.), pairs(2573, response_gain=2.)], SearchBudget(90000), actions)
    assert model is previous and len(agent.library.instrument_models) == 1
    assert not any(action["action"] == "instrument_mapping_counterexample" for action in actions)
    assert any(action["action"] == "reuse_instrument_mapping" for action in actions)
    call = agent._recall(COMMAND)[0]
    assert max(abs(call.predict(command) - row.target)
               for command, row in pairs(3319, response_gain=2.)) > .1


def test_meta_prior_without_measurement_roles_keeps_ordinary_control_transfer():
    library = KnowledgeLibrary()
    identity = Program("mul", children=(Program("feature", name="u"), Program("feature", name="u")))
    rows = [replace(command, target=identity.predict(command)) for command, _ in pairs(731)]
    source, _ = library.learn(Candidate("control", identity, "meta_prior_research"), COMMAND, rows)
    library.meta_priors.append({"source_concept_id": source, "source_features": ["u"]})
    agent = UnifiedResearchAgent(ResearchConfig(), library=library)
    agent._learn_instruments = agent._revise_inputs = True
    calls = agent._recall(COMMAND)
    assert calls and calls[0].predict(rows[3]) == identity.predict(rows[3])
