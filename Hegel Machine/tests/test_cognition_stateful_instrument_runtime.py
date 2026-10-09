"""Learned recurrent instruments remain replaceable inputs of measured calls."""

from dataclasses import replace
import json
from random import Random

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


COMMAND = TaskSpec("stateful", "Predict the response.", ("u",), (), (),
                   feature_bounds={"u": (-1., 1.)}, max_probe_length=96)
MEASURED = replace(COMMAND, feature_names=("u", "meter"))
METER = Program("feature", name="meter")
RESPONSE = Program("add", children=(Program("mul", children=(METER, METER)),
    Program("if", children=(Program("available", lag=1),
        Program("lag", name="meter", lag=1), Program("const", value=0.)))))


def pairs(seed, *, decay=.63, initial=.17, response_gain=1., length=64):
    rng, state, previous, history, controls, result = Random(seed), initial, 0., [], [], []
    for _ in range(length):
        u = rng.uniform(-1., 1.)
        state = decay * state + u
        reading = .25 * u + 1.2 * state
        target = response_gain * (reading ** 2 + previous)
        measured = Observation({"u": u, "meter": reading}, target, tuple(history))
        result.append((Observation({"u": u}, target, tuple(controls)), measured))
        history.append({**measured.features, "__target__": target})
        controls.append({"u": u, "__target__": target})
        previous = reading
    return result


def learned():
    library = KnowledgeLibrary()
    agent = UnifiedResearchAgent(ResearchConfig(max_history=4, minimum_improvement=1e-7), library=library)
    training = pairs(733)
    model = agent._fit_instruments(COMMAND, MEASURED, [training, pairs(977)], SearchBudget(90000), [])
    assert model and model["fields"]["meter"]["kind"] == "continuous_state"
    source, _ = library.learn(Candidate("measured_response", RESPONSE, "research"),
                              MEASURED, [row for _, row in training])
    library.meta_priors.append({"source_concept_id": source, "source_features": list(MEASURED.feature_names),
                               "control_features": list(COMMAND.feature_names), "object_schema": None})
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    agent = UnifiedResearchAgent(agent.config, library=restored)
    agent._learn_instruments = agent._revise_inputs = True
    return agent, source


def test_recurrent_instrument_rebinds_saved_mechanism_on_longer_new_trajectories():
    agent, source = learned()
    calls = agent._recall(COMMAND)
    assert calls
    for call in calls:
        assert call.definition.source_concept_id == source
        assert call.required_features() == {"u"}
        for seed, length in ((1709, 1), (2237, 96)):
            assert max(abs(call.predict(command) - measured.target)
                       for command, measured in pairs(seed, length=length)) < 1e-5


def test_state_drift_replaces_observation_model_and_retains_the_response_definition():
    agent, source = learned()
    change = {"decay": -.42, "initial": -.13}
    fresh = pairs(3371, **change)
    assert max(abs(agent._recall(COMMAND)[0].predict(command) - measured.target)
               for command, measured in fresh) > .1
    actions = []
    model = agent._fit_instruments(COMMAND, MEASURED,
        [pairs(1709, **change), pairs(2237, **change)], SearchBudget(90000), actions)
    assert model and model["fields"]["meter"]["kind"] == "continuous_state"
    assert any(action["action"] == "instrument_mapping_counterexample" for action in actions)
    assert any(old["status"] == "counterexample" for old in agent.library.instrument_models)
    for call in agent._recall(COMMAND):
        assert call.definition.source_concept_id == source
        assert max(abs(call.predict(command) - measured.target)
                   for command, measured in fresh) < 1e-5
    assert len(agent.library.concepts) == 1


def test_response_drift_does_not_revoke_supported_recurrent_readings():
    agent, _ = learned()
    previous = agent.library.instrument_models[-1]
    actions = []
    model = agent._fit_instruments(COMMAND, MEASURED,
        [pairs(1709, response_gain=1.8), pairs(2237, response_gain=1.8)],
        SearchBudget(90000), actions)
    assert model is previous
    assert len(agent.library.instrument_models) == 1
    assert not any(action["action"] == "instrument_mapping_counterexample" for action in actions)
    assert max(abs(agent._recall(COMMAND)[0].predict(command) - measured.target)
               for command, measured in pairs(3371, response_gain=1.8)) > .1
