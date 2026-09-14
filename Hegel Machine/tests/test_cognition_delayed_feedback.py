"""Actual delayed apparatus processes can be scheduled around and resumed."""

import json
import os
import signal
from time import monotonic, sleep

import pytest

from hegel_machine.cognition.delayed_feedback import DelayedFeedback
from hegel_machine.cognition.models import Probe


def wait_for(queue, ticket, timeout=5):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        result = queue.collect(ticket)
        if result is not None:
            return result
        sleep(.01)
    raise AssertionError("the apparatus did not complete its ticket")


def probe(name, rows=4):
    return Probe(name, tuple({"u": (index % 3 - 1) * .4} for index in range(rows)))


def test_another_task_finishes_while_delayed_experiment_survives_json_restore(tmp_path):
    queue = DelayedFeedback(tmp_path, observation_budget=12, measurement_budget=12)
    slow = queue.submit(probe("slow"), channels=("sensor_a",), delay_seconds=.6)
    fast = queue.submit(probe("fast"), task_index=2, channels=("sensor_a",))
    restored = DelayedFeedback.from_dict(json.loads(json.dumps(queue.to_dict())))
    result = wait_for(restored, fast)
    assert result["status"] == "succeeded" and len(result["observations"]) == 4
    assert restored.poll(slow)["status"] == "running"
    assert restored.costs == {"observation_cost": 4, "measurement_cost": 4, "experiment_cost": 1}
    assert restored.reserved == {"observation_cost": 4, "measurement_cost": 4, "experiment_cost": 1}
    result = wait_for(restored, slow)
    assert result["observations"][0]["history"] == []
    assert result["observations"][3]["history"][0]["sensor_a"] == result["observations"][0]["features"]["sensor_a"]
    again = DelayedFeedback.from_dict(json.loads(json.dumps(restored.to_dict())))
    assert again.collect(slow) is None and again.collect(fast) is None
    assert again.costs == {"observation_cost": 8, "measurement_cost": 8, "experiment_cost": 2}
    assert all(value == 0 for value in again.reserved.values())


def test_pending_reservations_prevent_parallel_overspend_and_cancel_releases_them(tmp_path):
    queue = DelayedFeedback(tmp_path, observation_budget=4, measurement_budget=4, experiment_budget=3)
    first = queue.submit(probe("reserved"), channels=("sensor_a",), delay_seconds=.3)
    with pytest.raises(ValueError, match="pending reservations"):
        queue.submit(probe("too_soon"), channels=("sensor_a",))
    queue.cancel(first)
    cancelled = wait_for(queue, first)
    assert cancelled["status"] == "cancelled" and cancelled["observations"] == []
    assert queue.costs == {"observation_cost": 0, "measurement_cost": 0, "experiment_cost": 1}
    second = queue.submit(probe("after_cancel"), channels=("sensor_a",))
    assert wait_for(queue, second)["status"] == "succeeded"
    before = queue.costs
    assert queue.cancel(second)["status"] == "succeeded"
    assert queue.collect(second) is None and queue.costs == before


def test_failed_real_apparatus_returns_error_once_and_counts_one_attempt(tmp_path):
    queue = DelayedFeedback(tmp_path)
    ticket = queue.submit(Probe("invalid_control", ({"u": 10.},)), channels=("sensor_a",))
    result = wait_for(queue, ticket)
    assert result["status"] == "failed" and "controllable input" in result["error"]
    assert queue.costs == {"observation_cost": 0, "measurement_cost": 0, "experiment_cost": 1}
    for _ in range(3):
        assert queue.collect(ticket) is None
        queue.poll(ticket)
    assert queue.costs["experiment_cost"] == 1
    assert all(value == 0 for value in queue.reserved.values())


def test_exited_process_is_a_collectable_failure_after_resume(tmp_path):
    queue = DelayedFeedback(tmp_path)
    ticket = queue.submit(probe("interrupted"), delay_seconds=2.)
    saved = json.loads(json.dumps(queue.to_dict()))
    os.kill(saved["tickets"][ticket]["process_id"], signal.SIGKILL)
    restored = DelayedFeedback.from_dict(saved)
    result = wait_for(restored, ticket)
    assert result["status"] == "failed" and result["error"]
    assert restored.costs["experiment_cost"] == 1
    assert restored.costs["observation_cost"] == 0
    assert restored.collect(ticket) is None
