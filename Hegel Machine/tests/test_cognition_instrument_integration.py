"""Learned instrument functions must survive calls, history, and session changes."""

from dataclasses import replace
import json

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession


CONFIG = ResearchConfig(observation_budget=384, compute_budget=360000,
    minimum_improvement=1e-7, max_rounds=6, active_probes=True, seed=221)


def purchased_pairs(lab, seed):
    probe = lab.random_probe(seed, length=32)
    measured = lab.acquire(probe, ("fine",)).observations
    history, commands = [], []
    for point, row in zip(probe.sequence, measured):
        commands.append(Observation(dict(point), row.target, tuple(history)))
        history.append({**point, "__target__": row.target})
    return list(zip(commands, measured))


def test_history_only_calls_keep_sensor_requirements_until_every_input_is_calibrated():
    library = KnowledgeLibrary()
    agent = UnifiedResearchAgent(CONFIG, library=library)
    with UnifiedObjectLab(objects=2, apparatus_seed=9013) as lab:
        measured_spec = lab.spec_for(("fine",))
        schema = json.loads(measured_spec.context["object_schema"])
        names = [schema["fields"][obj]["q"] for obj in schema["objects"]]
        first, second = purchased_pairs(lab, 183), purchased_pairs(lab, 885)
        agent._fit_instruments(lab.spec, measured_spec, [first, second], SearchBudget(24000), [])

        identity = Definition("instrument_history_identity", ("value",), Program("arg", name="value"))
        inner = Program("call", name=identity.definition_id, definition=identity,
            children=(Program("add", children=tuple(Program("lag", name=name, lag=1) for name in names)),))
        scaled = Definition("instrument_history_scaled", ("value",),
            Program("mul", children=(Program("const", value=2.), Program("arg", name="value"))))
        expression = Program("if", children=(Program("available", lag=1),
            Program("call", name=scaled.definition_id, definition=scaled, children=(inner,)),
            Program("const", value=0.)))
        assert expression.required_features() == set()
        assert expression.observed_variables() == {(name, 1) for name in names}
        evidence = [replace(row, target=expression.predict(row)) for _, row in first]
        concept_id, _ = library.learn(Candidate("historical_readout", expression, "readout"),
                                      measured_spec, evidence)
        library.meta_priors.append({"source_concept_id": concept_id,
            "source_features": list(measured_spec.feature_names),
            "control_features": list(measured_spec.feature_bounds), "object_schema": schema})

        restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
        learner = UnifiedResearchAgent(CONFIG, library=restored)
        learner._learn_instruments = learner._instrument_knowledge(lab)
        recalled = learner._recall(lab.spec_for(()))
        assert recalled
        fresh = purchased_pairs(lab, 1147)
        for program in recalled:
            assert program.required_features() == set()
            assert program.observed_variables()
            assert {name for name, _ in program.observed_variables()} <= set(lab.spec.feature_bounds)
            assert max(abs(program.predict(command) - expression.predict(measured))
                       for command, measured in fresh) < 1e-8

        # A history-only missing instrument is still a real measurement need.
        for model in restored.instrument_models:
            model["fields"].pop(names[1], None)
        assert learner._recall(lab.spec_for(())) == []
        assert learner._recall(measured_spec)


def test_session_restore_keeps_calibration_and_reopens_changed_event_instruments():
    tasks = [{"environment": "objects", "objects": 2, "readout": "event",
              "apparatus_seed": 9013, "seed": seed, "actuator_curvature": curvature,
              "rename_prefix": prefix}
             for seed, curvature, prefix in ((221, .12, ""), (331, .12, "again_"),
                                              (441, .31, "changed_"))]
    session = UnifiedResearchSession(CONFIG, tasks=tasks, measurement_budget=4608)
    first = session.run(max_tasks=1)
    assert first["knowledge"]["instrument_models"]
    assert first["results"][0]["research"]["selected_channels"] == []
    restored = UnifiedResearchSession(state=json.loads(json.dumps(first)))
    restored.run()
    report = restored.evaluate(length=48, trajectories=2, seed_offset=55013)
    assert all(row["loss"] < 1e-6 for row in report["evaluation"])
    same = report["results"][1]["research"]
    changed = report["results"][2]["research"]
    assert same["selected_channels"] == []
    assert same["costs"]["measurement_cost"] == 0
    assert changed["costs"]["measurement_cost"] > 0
    assert any(event["action"] == "instrument_mapping_counterexample" for event in changed["actions"])
    assert any(model["status"] == "counterexample" for model in report["knowledge"]["instrument_models"])
    assert any(model["status"] == "supported_on_measured_trajectory"
               for model in report["knowledge"]["instrument_models"])
