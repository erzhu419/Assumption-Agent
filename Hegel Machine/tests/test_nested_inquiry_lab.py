"""Historical nested physics can enter the common online research contract."""

from copy import deepcopy
from random import Random

import pytest

from hegel_machine.cognition.models import Probe
from scripts.run_nonlinear_mechanism_revision import laboratory


def task(**changes):
    return {"kind": "nested_response", "seed": 10112904, "apparatus_seed": 9912901,
        "response": "square", "inner_weight": .35, "window_width": 1,
        "noise_sigma": 0., **changes}


def trajectory(lab, n=5):
    return {"kind": "trajectory", "channels": [],
        "sequence": list(lab.random_probe(823, length=n).sequence)}


def test_public_adapter_preserves_historical_labels_and_control_only_history():
    legacy, inquiry = laboratory(task()), laboratory(task())
    action = trajectory(inquiry)
    original = legacy.acquire(Probe("legacy", tuple(action["sequence"])))
    record = inquiry.execute(action)
    measured = inquiry.observation_rows(record)
    assert [row.target for row in measured] == [row.target for row in original.observations]
    assert "__target__" in original.observations[-1].history[0]
    assert record["row_indices"] == list(range(5))
    assert record["measurements"] == []
    assert inquiry.costs == legacy.costs == {
        "observation_cost": 5, "measurement_cost": 0, "queries": 1}
    for index, row in enumerate(measured):
        assert row.features == action["sequence"][index]
        assert row.history == tuple(action["sequence"][:index])
    assert laboratory(task(response="base", apparatus_seed=17)).public_spec() == inquiry.public_spec()


def test_prospective_rows_do_not_buy_or_access_labels(monkeypatch):
    lab = laboratory(task())
    action = trajectory(lab)
    supplied = deepcopy(action)

    def unavailable(*args, **kwargs):
        raise AssertionError("prospective rows must not execute the apparatus")

    with monkeypatch.context() as context:
        context.setattr(lab, "acquire", unavailable)
        rows = lab.prediction_rows(action)
    assert lab.costs == {"observation_cost": 0, "measurement_cost": 0, "queries": 0}
    assert all(row.target == 0. for row in rows)
    measured = lab.observation_rows(lab.execute(action))
    assert [(row.features, row.history) for row in rows] == [
        (row.features, row.history) for row in measured]
    rows[-1].history[0]["x"] = 100.
    assert action == supplied
    assert lab.prediction_rows(action)[-1].history[0] == supplied["sequence"][0]


@pytest.mark.parametrize("noise_sigma", [0., .03])
def test_restore_preserves_paid_history_and_next_noise_without_reacquisition(monkeypatch, noise_sigma):
    case = task(noise_sigma=noise_sigma, window_width=3)
    original, restored = laboratory(case), laboratory(case)
    actions = original.confirmation_actions(67, Random(71))
    records = [original.execute(action) for action in actions]

    def unavailable(*args, **kwargs):
        raise AssertionError("restoration must not rerun paid physics")

    with monkeypatch.context() as context:
        context.setattr(restored, "acquire", unavailable)
        restored.restore_paid_account(records, original.costs)
    assert restored.costs == original.costs
    # Odd prefix length also checks Random.gauss' cached second draw.
    action = trajectory(original, 9)
    assert restored.execute(action) == original.execute(action)


def test_independent_confirmation_and_terminal_keep_public_horizons_and_paid_state():
    lab = laboratory(task(noise_sigma=.03, window_width=3))
    action = trajectory(lab)
    lab.execute(action)
    before = lab.costs
    assert [lab.action_cost(item) for item in lab.confirmation_actions(130, Random(19))] == [64, 64, 2]
    rows = lab.evaluation_rows(length=130, seed=947)
    assert [index for index, row in enumerate(rows) if not row.history] == [0, 64, 128]
    assert len(rows) == 130
    assert rows == lab.evaluation_rows(length=130, seed=947)
    assert rows != lab.evaluation_rows(length=130, seed=953)
    assert lab.costs == before
    assert lab.confirmation_precondition({"mse": .1}, mse_threshold=1e-6)
    assert lab.confirmation_precondition({"mse": 1e-10}, mse_threshold=1e-6) is None
    with pytest.raises(ValueError, match="exactly the public controls"):
        lab.execute({"kind": "trajectory", "sequence": [{"u": 0.}]})
    assert lab.costs == before


def test_common_trial_prefix_uses_two_public_controls(tmp_path):
    from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
    from scripts.run_anonymous_method_learning import trial_preparer

    case = task(prefix_seed=71, confirmation_seed=173)
    prepare = trial_preparer({}, [], tmp_path / "reference")
    payload = prepare(case, tmp_path / "trial",
        "def select_actions(context): return []\n",
        InquiryLLMConfig(observation_budget=256, confirmation_rows=96))
    agent, lab = payload["agent"], payload["laboratory"]
    assert agent.costs["observation_cost"] == lab.costs["observation_cost"] == 72
    assert [len(record["rows"]) for record in agent.observations] == [64, 8]
    assert all(set(row.features) == {"x", "z"} for row in agent.rows)
    assert agent.costs["model_calls"] == 0
