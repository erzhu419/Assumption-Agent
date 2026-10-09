"""Actual object input and relation changes share learned, serialized knowledge."""

import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession, laboratory_for
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


CONFIG = ResearchConfig(observation_budget=384, compute_budget=360000,
    minimum_improvement=1e-7, max_rounds=6, active_probes=True, seed=221)
BASE = {"environment": "objects", "objects": 2, "readout": "event",
        "apparatus_seed": 9013, "actuator_curvature": .12}


def serialize(value):
    return json.loads(json.dumps(value))


def continue_tasks(source, tasks):
    state = serialize(source)
    state["tasks"].extend(tasks)
    snapshots = []
    for _ in tasks:
        session = UnifiedResearchSession(state=state)
        state = serialize(session.run(max_tasks=1))
        snapshots.append(state)
    return snapshots


@pytest.fixture(scope="module")
def object_branches():
    # The complete source is learned from an empty library. No supplied law,
    # calibrated parameter, action return or source-library mutation is needed.
    source = serialize(UnifiedResearchSession(CONFIG,
        tasks=[{**BASE, "seed": 221}], measurement_budget=4608).run())
    assert source["results"][0]["research"]["confirmation_loss"] <= 1e-7
    assert source["knowledge"]["instrument_models"]
    changed_inputs = {**BASE, "seed": 441, "actuator_curvature": .31}
    changed_relation = {**BASE, "seed": 551, "relation_radius": 1.35}
    changed_both = {**changed_inputs, "seed": 661, "relation_radius": 1.35}
    branches = {
        "inputs": continue_tasks(source, [changed_inputs]),
        "relation": continue_tasks(source, [changed_relation]),
        "joint": continue_tasks(source, [changed_both, {**BASE, "seed": 771}]),
    }
    return source, branches


def research(state):
    return state["results"][-1]["research"]


def assert_fresh_response(state):
    result = research(state)
    task = state["tasks"][state["results"][-1]["task_position"]]
    with laboratory_for(task) as laboratory:
        fresh = laboratory.evaluation_rows(result["selected_channels"],
            seed=task["seed"] + 55013, length=48, trajectories=2)
        measured = loss(Program.from_dict(result["program"]), fresh,
                        result["fallback"], laboratory.spec.risk_quantile)
        # Alternative object scales are views of one instrument, rather than
        # channels that can be concatenated or charged twice.
        assert all(tuple(batch["channels"]) in laboratory.measurement_views
                   for batch in result["batches"])
    assert result["status"] == "completed", result["costs"]
    assert result["confirmation_loss"] <= 1e-7
    assert measured <= 1e-7, {"loss": measured, "kind": result["selected_kind"],
                             "costs": result["costs"]}


def assert_object_choice_and_common_response(state):
    result = research(state)
    choices = [item for item in result["actions"] if item["action"] == "choose_object_research"]
    assert choices
    assert any(len(item["options"]) > 1 for item in choices)
    feedback = [item for item in result["actions"]
                if item["action"] == "object_research_feedback"]
    measured = [item for item in feedback if item["before_loss"] is not None]
    assert measured
    task = state["tasks"][state["results"][-1]["task_position"]]
    with laboratory_for(task) as laboratory:
        for item in measured:
            batch = next(batch for batch in result["batches"]
                         if batch["probe"]["probe_id"] == item["probe_id"])

            def view_rows(channels):
                rows = [Observation(row["features"], row["target"], tuple(row["history"]))
                        for row in batch["observations"]]
                if channels:
                    return laboratory.project_rows(rows, batch["channels"], channels)
                history, commands = [], []
                for point, row in zip(batch["probe"]["sequence"], rows):
                    commands.append(Observation(point, row.target, tuple(history)))
                    history.append({**point, "__target__": row.target})
                return commands

            before = loss(Program.from_dict(item["baseline_program"]),
                view_rows(item["baseline_channels"]), item["baseline_fallback"], item["risk_quantile"])
            after = loss(Program.from_dict(item["program"]), view_rows(item["channels"]),
                         item["fallback"], item["risk_quantile"])
            assert item["before_loss"] == pytest.approx(before, rel=1e-12, abs=1e-12)
            assert item["after_loss"] == pytest.approx(after, rel=1e-12, abs=1e-12)
            assert item["observation_rows"] == len(batch["observations"])
            assert item["trial"] is not None
    # Repairing fields supplies a proposal. The action becomes supported only
    # when both old and revised responses have been checked on new output data.
    assert any(item["status"] == "supported" and item["confirmed"]
               and item["after_loss"] <= 1e-7 for item in measured)
    assert all(item["before_loss"] is not None for item in feedback if item["status"] == "supported")


def test_input_change_recovers_the_learned_object_response_after_json(object_branches):
    source, branches = object_branches
    state = branches["inputs"][0]
    assert state["results"][0] == source["results"][0]
    assert_fresh_response(state)
    assert_object_choice_and_common_response(state)
    assert any(item["action"] == "instrument_mapping_counterexample"
               for item in research(state)["actions"])
    # Original callable knowledge remains available while its dependency is
    # revised; an input failure is not permission to erase the source concept.
    original = {item["concept_id"] for item in source["knowledge"]["concepts"]}
    assert original <= {item["concept_id"] for item in state["knowledge"]["concepts"]}


def test_relation_change_keeps_correct_instrument_knowledge(object_branches):
    source, branches = object_branches
    state = branches["relation"][0]
    assert_fresh_response(state)
    assert_object_choice_and_common_response(state)
    before = {item["model_id"]: item for item in source["knowledge"]["instrument_models"]
              if item["status"] == "supported_on_measured_trajectory"}
    after = {item["model_id"]: item for item in state["knowledge"]["instrument_models"]}
    assert before
    for model_id, model in before.items():
        assert after[model_id]["status"] != "counterexample"
        assert {name: field["program"] for name, field in after[model_id]["fields"].items()} == {
            name: field["program"] for name, field in model["fields"].items()}
    assert not any(item["action"] == "instrument_mapping_counterexample"
                   for item in research(state)["actions"])


def test_joint_change_and_return_survive_full_json_without_confusing_input_and_response(object_branches):
    source, branches = object_branches
    changed, returned = branches["joint"]
    # The second continuation inherits the complete real action and instrument
    # history, rather than a selected concept or a hand-written policy score.
    assert returned["results"][:-1] == changed["results"]
    assert_fresh_response(changed)
    assert_fresh_response(returned)
    assert_object_choice_and_common_response(changed)
    result = research(changed)
    actions = result["actions"]
    boundary_steps = [index for index, item in enumerate(actions)
                      if item["action"] == "acquire" and item["purpose"] in {
                          "distinguish_program_conditions", "distinguish_inherited_relation"}]
    confirmed = next(index for index, item in enumerate(actions)
                     if item["action"] == "object_research_feedback" and item["status"] == "supported")
    # Input repair can expose a new relation interval. That new question must
    # be tested before a short random confirmation can mark the visit solved.
    assert boundary_steps and max(boundary_steps) < confirmed
    tested = []
    task = changed["tasks"][changed["results"][-1]["task_position"]]
    with laboratory_for(task) as laboratory:
        rates = {channel.name: channel.cost_per_row for channel in laboratory.channels}
        for index in boundary_steps:
            acquisition = actions[index]
            batch = next(batch for batch in result["batches"]
                         if batch["probe"]["probe_id"] == acquisition["probe_id"])
            check_index, check = next((position, item) for position, item in enumerate(actions)
                if item["action"] in {"test_competing_program_predictions", "test_inherited_relation_predictions"}
                and item["probe_id"] == acquisition["probe_id"])
            assert index < check_index < confirmed
            assert acquisition["observation_cost"] == len(batch["observations"]) > 0
            assert acquisition["measurement_cost"] == len(batch["observations"]) * sum(
                rates[channel] for channel in batch["channels"])
            assert check["evidence"]["observation_rows"] == len(batch["observations"])
            tested.append(check)
    assert any(item["plan"]["predicted_separation"] > CONFIG.minimum_improvement
               and item["evidence"]["status"] == "distinguished" for item in tested)
    assert any(item["action"] == "instrument_mapping_counterexample"
               for item in actions)
    original = {item["concept_id"] for item in source["knowledge"]["concepts"]}
    assert original <= {item["concept_id"] for item in returned["knowledge"]["concepts"]}


def test_fresh_structure_failure_reopens_research_with_newly_paid_evidence(monkeypatch):
    observed_fits = []
    research_view = UnifiedResearchAgent._research_view

    def record_training(self, spec, training, validation, *args, **kwargs):
        if "measured_q_o0" in spec.feature_names:
            observed_fits.append((len(training), len(validation)))
        return research_view(self, spec, training, validation, *args, **kwargs)

    monkeypatch.setattr(UnifiedResearchAgent, "_research_view", record_training)
    config = ResearchConfig(observation_budget=384, compute_budget=360000,
        minimum_improvement=1e-7, max_rounds=6, active_probes=True, seed=23004)
    base = {"environment": "objects", "apparatus_seed": 22003, "profile": "development"}
    tasks = [{**base, "readout": "total", "seed": 23004},
             {**base, "readout": "event", "seed": 33011}]
    session = UnifiedResearchSession(config, tasks=tasks, measurement_budget=4608)
    report = session.run()
    result = research(report)
    feedback = [item for item in result["actions"] if item["action"] == "object_research_feedback"
                and item["direction"] == "research_object_structure" and item["observation_rows"]]
    assert feedback[0]["status"] == "refuted"
    assert feedback[-1]["status"] == "supported"
    assert feedback[0]["probe_id"] != feedback[-1]["probe_id"]
    assert observed_fits[0][0] == 8
    assert any(training >= 32 and validation > 0 for training, validation in observed_fits[1:])
    assert result["concepts_added"]
    assert result["costs"]["observation_cost"] <= 384
    assert result["costs"]["measurement_cost"] <= 4608
    assert result["costs"]["compute_cost"] <= 360000
    assert_fresh_response(report)
