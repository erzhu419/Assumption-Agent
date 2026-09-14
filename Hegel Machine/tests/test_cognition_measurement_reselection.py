"""New paid trajectory failures reopen instruments and survive process restart."""

from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import make_lab
from hegel_machine.cognition.measurement_research import InstrumentedEnvironment, MeasurementResearchAgent
from hegel_machine.cognition.measurement_session import MeasurementSession
from hegel_machine.cognition.models import Probe, ResearchConfig


def _config(**changes):
    return replace(ResearchConfig(observation_budget=128, compute_budget=100000, max_rounds=3, seed=313),
                   **changes)


@pytest.mark.parametrize("steps", [0, -1])
def test_invalid_research_step_limit_spends_nothing(steps, monkeypatch):
    with make_lab(3, 313) as laboratory:
        with pytest.raises(ValueError, match="research steps"):
            MeasurementResearchAgent(_config()).run(laboratory, max_steps=steps)
        assert not laboratory.acquisitions
    def no_apparatus(*args, **kwargs):
        pytest.fail("An invalid research-step limit must be rejected before opening an apparatus")
    monkeypatch.setattr("hegel_machine.cognition.measurement_session.make_lab", no_apparatus)
    session = MeasurementSession(_config(), tasks=[3])
    with pytest.raises(ValueError, match="research steps"):
        session.run(research_steps=steps)
    assert session.run(max_steps=0)["total_observation_cost"] == 0


@pytest.mark.parametrize("backend", ["local", "process"])
def test_apparatus_restore_rebuilds_old_experiments_without_new_paid_evidence(backend):
    warmup = Probe("operating", tuple({"u": .3 if i % 2 else -.5} for i in range(48)))
    later = Probe("later", tuple({"u": -.2 + .05 * i} for i in range(8)))
    fresh = Probe("next", tuple({"u": .4 - .05 * i} for i in range(8)))
    with make_lab(3, 313, backend=backend) as original:
        original.acquire(warmup, ("sensor_a",))
        before = original.acquire(later, ("sensor_a",)).observations
        commands = json.loads(json.dumps(original.acquisitions))
        paid = dict(original.costs)
        with make_lab(3, 313, backend=backend) as restored:
            restored.restore(commands)
            assert restored.costs == paid
            measured = restored.acquire(later, ("sensor_a", "sensor_b")).observations
            assert [row.target for row in measured] == [row.target for row in before]
            assert all("sensor_b" not in row.features for row in before)
            assert any(abs(row.features["sensor_b"]) > 0 for row in measured)
            assert restored.costs["observation_cost"] == paid["observation_cost"] + 8
            assert restored.costs["measurement_cost"] == paid["measurement_cost"] + 40
            original.acquire(later, ("sensor_a", "sensor_b"))
            assert original.acquire(fresh, ("sensor_a", "sensor_b")) == restored.acquire(fresh, ("sensor_a", "sensor_b"))


def test_later_residual_buys_an_instrument_and_beats_keeping_the_initial_view():
    config = _config()
    with make_lab(3, 313) as adaptive_lab:
        agent = MeasurementResearchAgent(config, measurement_budget=512)
        initial = agent.run(adaptive_lab, max_steps=1)
        assert initial["selected_channels"] == ["sensor_a"]
        assert initial["research_state"]["view_validation_loss"] < 1e-10
        assert all(row["features"]["sensor_b"] == 0 for batch in initial["pilot_observations"] for row in batch)
        completed = agent.run(adaptive_lab, research_state=json.loads(json.dumps(initial["research_state"])))
        event = completed["measurement_events"][0]
        assert event["reason"] == "new_trajectory_residual" and event["adopted"]
        assert event["current_validation_loss"] > event["previous_validation_loss"] + config.minimum_improvement
        assert completed["selected_channels"] == ["sensor_a", "sensor_b"]
        assert event["costs"]["compute_cost"] == sum(item["compute_cost"] for item in event["selection"])
        assert all(event["costs"][key] > 0 for key in ("observation_cost", "measurement_cost", "compute_cost"))
        rows = adaptive_lab.evaluation_rows(completed["selected_channels"], seed=1441)
        learned_loss = loss(Program.from_dict(completed["episode"]["program"]), rows,
                            completed["episode"]["fallback"], .5)
        assert learned_loss < 1e-8
        state = completed["research_state"]
        # Historical trajectories without the new sensor remain explicitly
        # incomplete. They are not filled with latent readings or predictions.
        incomplete = [batch for batch in state["batches"] if batch["channels"] == ["sensor_a"]]
        assert incomplete and all("sensor_b" not in row["features"] for batch in incomplete for row in batch["observations"])
        assert all("sensor_b" in row["features"] for row in state["workspace"]["observations"])
        complete_rows = sum(len(batch["observations"]) for batch in state["batches"]
                            if set(completed["selected_channels"]) <= set(batch["channels"]))
        assert len(state["workspace"]["observations"]) == complete_rows
        assert len(state["episodes"]) == 2
        assert state["episodes"][1]["actions"][0]["action"] == "resume_research"
        assert any(action["action"] == "revise" for action in state["episodes"][1]["actions"])
        assert completed["costs"]["observation_cost"] == sum(query["observation_cost"] for query in adaptive_lab.acquisitions)
        assert completed["costs"]["measurement_cost"] == sum(query["measurement_cost"] for query in adaptive_lab.acquisitions)

    # The fixed-view arm gets the same initial data and the entire remaining
    # observation/compute allowance, then continues ordinary program research.
    with make_lab(3, 313) as fixed_lab:
        fixed_lab.restore(initial["research_state"]["apparatus_acquisitions"])
        fixed = CognitiveAgent(replace(config,
            observation_budget=config.observation_budget - initial["costs"]["observation_cost"],
            compute_budget=config.compute_budget - initial["costs"]["compute_cost"])).run(
                InstrumentedEnvironment(fixed_lab, initial["selected_channels"]),
                research_state=initial["research_state"]["workspace"])
        fixed_rows = fixed_lab.evaluation_rows(initial["selected_channels"], seed=1441)
        assert loss(fixed.program, fixed_rows, fixed.fallback, .5) > .03
        assert all(query["channels"] == ["sensor_a"] for query in fixed_lab.acquisitions[2:])
        assert fixed_lab.costs["observation_cost"] <= config.observation_budget


def test_mid_question_json_resume_reopens_the_same_process_and_keeps_knowledge(tmp_path):
    saved = []
    session = MeasurementSession(_config(), tasks=[3], backend="process", measurement_budget=512)
    paused = session.run(max_steps=1, research_steps=2, on_checkpoint=saved.append)
    assert paused["status"] == "paused" and paused["completed_tasks"] == []
    assert paused["pending_tasks"] == [3] and paused["active_task"]["task_index"] == 3
    state = paused["active_task"]["result"]["research_state"]
    assert state["stage"] == "reselect" and state["measurement_question"]
    assert paused["knowledge"]["concepts"]
    assert all(not checkpoint["results"] for checkpoint in saved)
    assert "evaluation" not in paused["active_task"]["result"]
    path = tmp_path / "measurements.json"
    path.write_text(json.dumps(paused))
    restored = MeasurementSession(state=json.loads(path.read_text()))
    assert restored.library.to_dict() == paused["knowledge"]
    middle = restored.run(max_steps=1, research_steps=1)
    assert middle["status"] == "paused" and middle["completed_tasks"] == []
    selected = middle["active_task"]["result"]
    assert selected["selected_channels"] == ["sensor_a", "sensor_b"]
    assert selected["research_state"]["stage"] == "research"
    assert middle["total_observation_cost"] > paused["total_observation_cost"]
    again = MeasurementSession(state=json.loads(json.dumps(middle)))
    final = again.run(max_steps=1)
    assert final["status"] == "completed" and final["completed_tasks"] == [3]
    assert final["active_task"] is None and not final["pending_tasks"]
    result = final["results"][0]
    assert result["evaluation"]["rows"] == 256 and result["evaluation"]["mse"] < 1e-8
    assert result["pilot_observations"] == selected["pilot_observations"]
    commands = result["research_state"]["apparatus_acquisitions"]
    assert len([query for query in commands if query["probe_id"].startswith("instrument_pilot_")
                and query["probe_id"] in {"instrument_pilot_0", "instrument_pilot_1"}]) == 2
    for kind, limit in (("observation", 128), ("measurement", 512), ("compute", 100000)):
        assert final[f"total_{kind}_cost"] == result["costs"][f"{kind}_cost"] <= limit
    old_ids = {item["concept_id"] for item in paused["knowledge"]["concepts"]}
    assert old_ids <= {item["concept_id"] for item in final["knowledge"]["concepts"]}


def test_unfunded_measurement_question_stays_saved_without_reading_missing_channels():
    with make_lab(3, 313) as laboratory:
        result = MeasurementResearchAgent(_config(observation_budget=96, compute_budget=60000),
                                           measurement_budget=320).run(laboratory)
        assert result["status"] == "budget_exhausted"
        assert result["selected_channels"] == ["sensor_a"]
        assert result["measurement_question"]["status"] == "awaiting_budget"
        assert result["research_state"]["stage"] == "reselect"
        assert not result["measurement_events"]
        assert all(query["channels"] == ["sensor_a"] for query in laboratory.acquisitions[2:])
        assert result["costs"]["measurement_cost"] <= 320
        saved = json.loads(json.dumps(result["research_state"]))
    with make_lab(3, 313, backend="process") as resumed_lab:
        resumed = MeasurementResearchAgent(_config(observation_budget=96, compute_budget=60000),
                                            measurement_budget=320).run(resumed_lab, research_state=saved)
        assert resumed["status"] == "budget_exhausted"
        assert resumed["costs"] == result["costs"]
        assert resumed["measurement_question"] == result["measurement_question"]


def test_saved_unfunded_question_finishes_after_actual_budget_extension(tmp_path):
    limited = _config(observation_budget=96, compute_budget=60000)
    session = MeasurementSession(limited, tasks=[3], backend="process", measurement_budget=320)
    stopped = session.run()
    assert stopped["status"] == "budget_exhausted" and stopped["pending_tasks"] == [3]
    waiting = stopped["active_task"]["result"]
    assert waiting["measurement_question"]["status"] == "awaiting_budget"
    old_commands = waiting["research_state"]["apparatus_acquisitions"]
    path = tmp_path / "unfunded_measurement.json"
    path.write_text(json.dumps(stopped))
    saved = json.loads(path.read_text())
    resumed = MeasurementSession(_config(), measurement_budget=512, state=saved)
    final = resumed.run()
    assert final["status"] == "completed" and final["completed_tasks"] == [3]
    result = final["results"][0]
    assert result["selected_channels"] == ["sensor_a", "sensor_b"]
    assert result["measurement_events"][0]["adopted"]
    assert result["measurement_events"][0]["status"] == "completed"
    assert result["measurement_events"][0]["remaining"]["measurement"] == 512 - stopped["total_measurement_cost"]
    assert result["evaluation"]["mse"] < 1e-8
    commands = result["research_state"]["apparatus_acquisitions"]
    assert commands[:len(old_commands)] == old_commands
    assert len([query for query in commands if query["probe_id"] in
                {"instrument_pilot_0", "instrument_pilot_1"}]) == 2
    assert final["config"]["observation_budget"] == 128
    assert final["config"]["compute_budget"] == 100000
    assert final["measurement_budget"] == 512
    for kind, limit in (("observation", 128), ("measurement", 512), ("compute", 100000)):
        assert stopped[f"total_{kind}_cost"] < final[f"total_{kind}_cost"] <= limit
    for changed in (replace(limited, seed=314), replace(limited, max_rounds=4),
                    replace(limited, observation_budget=95)):
        with pytest.raises(ValueError, match="resume"):
            MeasurementSession(changed, state=saved)
    with pytest.raises(ValueError, match="resume"):
        MeasurementSession(measurement_budget=319, state=saved)


@pytest.mark.parametrize("task_index", range(3))
def test_stable_apparatus_does_not_reopen_an_already_sufficient_instrument(task_index):
    seed = 313 + task_index * 10007
    with make_lab(task_index, seed) as laboratory:
        result = MeasurementResearchAgent(_config(seed=seed), measurement_budget=512).run(laboratory)
        assert result["status"] == "completed"
        assert result["selected_channels"] == ["sensor_a"]
        assert not result["measurement_events"] and not result["measurement_question"]
        assert all(query["channels"] == ["sensor_a"] for query in laboratory.acquisitions[2:])
