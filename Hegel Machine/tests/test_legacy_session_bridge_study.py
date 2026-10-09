"""The session study must isolate knowledge and retain actual failure outcomes."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_legacy_session_bridge_study as study


def test_registered_cohort_keeps_twelve_distinct_apparatuses_and_meter_development():
    plan = study.protocol()
    assert len(plan["cases"]) == len({case["apparatus_seed"] for case in plan["cases"]}) == 12
    assert plan["development"]["apparatus_seed"] not in {case["apparatus_seed"] for case in plan["cases"]}
    assert plan["development"]["consumer_channels"] == ["meter"]
    assert all(case["consumer_channels"] == [] for case in plan["cases"])
    assert plan["limits"]["confirmation_rows"] == 24
    assert plan["external_model_calls"] == 0


def test_cloned_paid_episode_does_not_purchase_or_lose_measurements(tmp_path):
    plan = study.protocol()
    episode = study.make_episode(tmp_path / "original" / "consumer", plan["development"], plan, "consumer")
    agent, lab = episode["agent"], episode["laboratory"]
    agent._execute("query", {"actions": [{"kind": "trajectory", "channels": ["meter"],
        "sequence": [{"u": .2}, {"u": -.4}, {"u": .7}]}]}, lab)
    agent._save("shared/example.py", "def predict(row): return 0.\n")
    previous = deepcopy(agent.result("paid"))
    cloned = study.clone_episode(episode, tmp_path / "cloned" / "consumer")
    copied = cloned["agent"]
    assert copied.costs == agent.costs == previous["costs"]
    assert copied.observations == agent.observations
    assert copied.rows == agent.rows
    assert cloned["laboratory"].costs == lab.costs
    assert copied.workspace != agent.workspace and copied.workspace.name == agent.workspace.name
    assert json.loads((copied.workspace / "laboratory_observations.json").read_text()) == json.loads(json.dumps(copied.observations))
    assert (copied.workspace / "shared/example.py").read_text() == agent.files["shared/example.py"]
    copied._execute("query", {"actions": [{"kind": "trajectory", "channels": [],
        "sequence": [{"u": -.1}, {"u": .5}]}]}, cloned["laboratory"])
    assert copied.costs["observation_cost"] == 5
    assert copied.costs["measurement_cost"] == 9
    assert agent.result("paid") == previous
    assert lab.costs["observation_cost"] == 3


def test_entire_registered_roster_trains_before_scoring_even_with_failure(tmp_path, monkeypatch):
    output = tmp_path / "study"
    events = []
    registered = study.protocol()

    def train(case, path, plan):
        assert json.loads((output / "protocol.json").read_text()) == registered
        events.append(("train", case["apparatus_seed"]))
        if case == plan["cases"][2]:
            raise RuntimeError("record this real execution failure")
        return {"case": case, "output": path}

    def score(case, plan):
        assert len([event for event in events if event[0] == "train"]) == 12
        events.append(("score", case["case"]["apparatus_seed"]))
        return {"seed": case["case"]["apparatus_seed"]}

    def summarize(records, failures, count):
        return {"registered_cases": count, "completed_cases": len(records), "execution_failures": failures}

    monkeypatch.setattr(study, "train_case", train)
    monkeypatch.setattr(study, "score_case", score)
    monkeypatch.setattr(study, "summarize", summarize)
    monkeypatch.setattr(sys, "argv", ["study", "--output", str(output), "--phase", "cohort"])
    study.main()
    summary = json.loads((output / "cohort" / "summary.json").read_text())
    assert summary["registered_cases"] == 12 and summary["completed_cases"] == 11
    assert len(summary["execution_failures"]) == 1
    assert "record this real execution failure" in summary["execution_failures"][0]["error"]
    assert [event[0] for event in events[:12]] == ["train"] * 12


def test_summary_does_not_count_workflow_support_as_confirmed_return():
    empty = {"tasks": [{"confirmed_and_correct": False}, {"confirmed_and_correct": True}],
        "costs": dict(zip(study.COSTS, (200, 300, 400))),
        "incremental_costs": dict(zip(study.COSTS, (24, 0, 96))),
        "revisits": [{"status": "completed"}], "confirmed_positive_source_returns": 0,
        "workflow_knowledge_outcomes": [{"shared_gain": 1., "source_returns": []}],
        "source_confirmation_statuses": {"provisional": 1}, "negative_source_returns": 0,
        "measurement_required_export_failures": 1,
        "export_failures_by_reason": {"temporal predictors can read only control u": 1},
        "external_model_calls": 0}
    result = study.summarize([{"results": {arm: deepcopy(empty) for arm in study.ARMS}}], [], 1)
    for arm in study.ARMS:
        assert result["arms"][arm]["consumer_confirmed_and_correct"] == 0
        assert result["arms"][arm]["confirmed_positive_source_returns"] == 0
        assert result["arms"][arm]["workflow_knowledge_outcomes"] == 1
        assert result["arms"][arm]["source_confirmation_statuses"] == {"provisional": 1}
        assert result["arms"][arm]["measurement_required_export_failures"] == 1
    assert result["paired_retained_minus_removed"] == dict.fromkeys(study.COSTS, 0)


def test_real_fork_changes_only_knowledge_before_native_planning(tmp_path, monkeypatch):
    def forbidden_terminal(*args, **kwargs):
        pytest.fail("terminal observations requested during learning")

    monkeypatch.setattr(study.TemporalInquiryLab, "evaluation_rows", forbidden_terminal)
    plan = study.protocol()
    trained = study.train_case(plan["development"], tmp_path / "case", plan)
    retained, removed = (trained["arms"][arm] for arm in study.ARMS)
    initial = [deepcopy(payload["initial"]) for payload in (retained, removed)]
    retained_library, removed_library = (state.pop("knowledge") for state in initial)
    assert retained_library["concepts"] and not removed_library["concepts"]
    assert initial[0] == initial[1]
    assert not initial[0]["pending_revisits"] and not initial[0]["evaluation"]
    assert trained["source_added_concepts"]
    for payload in (retained, removed):
        for position, episode in enumerate(payload["episodes"]):
            agent = episode["agent"]
            assert agent.model is None and not agent.confirmations
            assert agent.costs["model_calls"] == 0
            assert agent.costs["observation_cost"] <= agent.config.observation_budget - agent.config.confirmation_rows
            if position == 0:
                assert len(agent.observations[:3]) == 3
                assert sum(len(item["rows"]) for item in agent.observations[:3]) == 72
    assert retained["episodes"][0]["agent"].observations[:3] == removed["episodes"][0]["agent"].observations[:3]
