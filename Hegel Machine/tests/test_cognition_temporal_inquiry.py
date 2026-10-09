"""Temporal adapter semantics, independent of any model-provider performance."""

from random import Random

import pytest

from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab


def trajectory(values, *, measured=False):
    return {"kind": "trajectory", "sequence": [{"u": value} for value in values],
            "channels": ["meter"] if measured else []}


def test_short_source_cannot_distinguish_hidden_instrument_worlds():
    finite = TemporalInquiryLab(apparatus_seed=739, instrument_mode="finite2")
    recurrent = TemporalInquiryLab(apparatus_seed=739, instrument_mode="recurrent")
    assert finite.public_spec() == recurrent.public_spec()
    assert not any(key in finite.public_spec() for key in ("seed", "task_index", "instrument_mode", "finite_language"))
    rng = Random(151)
    for _ in range(4):
        action = trajectory([rng.uniform(-1, 1) for _ in range(3)], measured=True)
        left, right = finite.execute(action), recurrent.execute(action)
        assert [row["target"] for row in left["rows"]] == pytest.approx(
            [row["target"] for row in right["rows"]], abs=1e-12)
        assert [row["values"]["meter"] for row in left["measurements"]] == pytest.approx(
            [row["values"]["meter"] for row in right["measurements"]], abs=1e-12)


def test_long_trajectory_exposes_hidden_world_difference():
    finite = TemporalInquiryLab(task_index=1, apparatus_seed=739, instrument_mode="finite2")
    recurrent = TemporalInquiryLab(task_index=1, apparatus_seed=739, instrument_mode="recurrent")
    action = trajectory([1., 0., 0., 0., 0., 0.], measured=True)
    left, right = finite.execute(action), recurrent.execute(action)
    assert left["measurements"][3]["values"]["meter"] == 0.
    assert abs(right["measurements"][3]["values"]["meter"]) > .01
    assert abs(left["rows"][3]["target"] - right["rows"][3]["target"]) > .001


def test_purchased_meter_stays_separate_from_free_running_prediction_inputs():
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=739)
    action = trajectory([.8, -.2, 0., .4], measured=True)
    assert lab.action_cost(action) == 4
    assert lab.costs == {"observation_cost": 0, "measurement_cost": 0, "queries": 0}
    record = lab.execute(action)
    assert record["row_indices"] == [0, 1, 2, 3]
    assert record["observation_cost"] == 4
    assert record["measurement_cost"] == 12
    assert len(record["measurements"]) == 4
    for index, row in enumerate(record["rows"]):
        assert row["features"] == action["sequence"][index]
        assert list(row["history"]) == action["sequence"][:index]
    repeated = lab.execute(trajectory([.8, -.2, 0., .4]))
    assert repeated["rows"] == record["rows"]
    assert repeated["measurements"] == []
    assert repeated["observation_index"] == 4
    assert repeated["row_indices"] == [4, 5, 6, 7]
    assert lab.costs == {"observation_cost": 8, "measurement_cost": 12, "queries": 2}


@pytest.mark.parametrize("stage,horizon", [(0, 3), (1, 24)])
def test_confirmation_and_terminal_cover_horizon_with_exact_row_count(stage, horizon):
    lab = TemporalInquiryLab(task_index=stage, apparatus_seed=739)
    actions = lab.confirmation_actions(2 * horizon + 1, Random(173))
    assert [lab.action_cost(action) for action in actions] == [horizon, horizon, 1]
    assert all(action["channels"] == [] for action in actions)
    rows = lab.evaluation_rows(length=2 * horizon + 1, seed=173)
    assert len(rows) == 2 * horizon + 1
    assert [len(row.history) for row in rows] == [*range(horizon), *range(horizon), 0]
    assert all(set(row.features) == {"u"} and all(set(point) == {"u"} for point in row.history)
               for row in rows)
    assert lab.costs == {"observation_cost": 0, "measurement_cost": 0, "queries": 0}
    assert rows == lab.evaluation_rows(length=2 * horizon + 1, seed=173)


@pytest.mark.parametrize("action", [
    trajectory([]), trajectory([0.] * 4), trajectory([1.1]),
    {"kind": "response", "point": {"u": 0.}},
    {"kind": "trajectory", "sequence": [{"u": 0.}], "channels": ["unknown"]},
    {"kind": "trajectory", "sequence": [{"meter": 0.}], "channels": []},
])
def test_invalid_purchases_are_rejected_before_acquisition(action):
    lab = TemporalInquiryLab()
    with pytest.raises(ValueError):
        lab.action_cost(action)
    with pytest.raises(ValueError):
        lab.execute(action)
    assert lab.costs == {"observation_cost": 0, "measurement_cost": 0, "queries": 0}


def test_shared_agent_prices_rows_and_confirms_on_long_control_histories(tmp_path):
    import json
    from hegel_machine.cognition.online_inquiry_llm import InquiryLLMAgent, InquiryLLMConfig

    lab = TemporalInquiryLab(task_index=1, apparatus_seed=739)
    agent = InquiryLLMAgent(None, tmp_path, arm="code_react",
        config=InquiryLLMConfig(observation_budget=54, confirmation_rows=48))
    result = agent._query({"actions": [trajectory([1., 0., 0., 0., 0., 0.], measured=True)],
                          "question": {"id": "tail", "question": "Does the response persist?"}}, lab)
    assert result["observation_cost"] == agent.costs["observation_cost"] == 6
    assert agent.costs["measurement_cost"] == 18
    assert result["question"]["evidence"] == list(range(6))
    denied = agent._query({"actions": [trajectory([0.] * 24)] * 3}, lab)
    assert denied["error"] == "observation budget exhausted"
    assert lab.costs["observation_cost"] == 6
    (tmp_path / "model.py").write_text("def predict(row): return 0.0\n")
    failed = agent._execute("finish", {"model": "model.py"}, lab)["fresh_confirmation"]
    assert not failed["passed"]
    assert failed["rows"] == 48
    assert failed["observation_indices"] == list(range(6, 54))
    assert [len(record["rows"][-1]["history"]) for record in failed["observations"]] == [23, 23]
    assert agent.costs["observation_cost"] == lab.costs["observation_cost"] == 54
    assert len(agent.rows) == 54
    rows = json.loads((tmp_path / "model_observations.json").read_text())
    assert all(set(row["features"]) == {"u"} and all(set(p) == {"u"} for p in row["history"])
               for row in rows)


def test_temporal_protocol_keeps_matched_short_sources_and_ample_confirmation():
    from scripts.run_temporal_inquiry_llm import laboratory, protocol

    plan = protocol(arms=["code_react"])
    assert plan["arms"] == ["code_react"]
    assert plan["limits"]["confirmation_rows"] >= 24
    left, right = plan["cases"]
    action = trajectory([.1, -.6, .7], measured=True)
    l = laboratory(left["source"]["lab"]).execute(action)
    r = laboratory(right["source"]["lab"]).execute(action)
    assert [row["target"] for row in l["rows"]] == pytest.approx([row["target"] for row in r["rows"]])
    assert left["source"]["evaluation_seed"] == right["source"]["evaluation_seed"]
    assert laboratory(left["consumers"][0]["lab"]).public_spec()["max_probe_length"] == 24
