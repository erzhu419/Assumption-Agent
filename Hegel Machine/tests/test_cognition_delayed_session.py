import json
import time

from hegel_machine.cognition.delayed_feedback import DelayedFeedback
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.models import Probe, ResearchConfig
from hegel_machine.cognition.session import ResearchSession


def test_waiting_task_releases_scheduler_and_receives_saved_experiment_once(tmp_path):
    config = ResearchConfig(observation_budget=24, compute_budget=5000, max_rounds=2, seed=17)
    environments = make_suite(17)[:2]
    session = ResearchSession(config, environments, observation_budget=100, compute_budget=20000,
                              feedback=DelayedFeedback(tmp_path, observation_budget=100), max_visits=1)
    probe = Probe("delayed", ({"x": -.5, "z": 0.}, {"x": .2, "z": 0.}, {"x": .8, "z": 0.}))
    ticket = session.submit_experiment(0, probe, delay_seconds=.8)
    assert session._remaining()["observation_budget"] == 97
    result = session.run(max_steps=1)
    assert result["episodes"][0]["task_index"] == 1
    restored = ResearchSession(config, environments, observation_budget=100, compute_budget=20000,
                               max_visits=1, state=json.loads(json.dumps(result)))
    deadline = time.monotonic() + 5
    # The ordinary task may outlast the apparatus delay, in which case run()
    # already delivered the result before writing its checkpoint.
    received = [event for event in restored.state["scheduler_events"]
                if event.get("reason") == "experiment_delivered" and event.get("ticket_id") == ticket]
    while not received and time.monotonic() < deadline:
        received = restored.poll_experiments()
        if not received:
            time.sleep(.02)
    assert received[0]["ticket_id"] == ticket and received[0]["status"] == "succeeded"
    before = restored.state["total_observation_cost"]
    assert len(restored.state["workspaces"]["0"]["observations"]) == 3
    assert not restored.poll_experiments()
    assert restored.state["total_observation_cost"] == before
    continued = restored.run()
    assert continued["status"] == "completed"
    episode = continued["episodes"][-1]
    assert episode["task_index"] == 0
    assert any(action["action"] == "resume_research" for action in episode["actions"])
    assert continued["total_experiment_cost"] == 1
