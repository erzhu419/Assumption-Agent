"""Physical ambiguity and shared-runtime integration; no discovery claims."""

from copy import deepcopy
from itertools import product

import pytest

from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from test_common_archive_online_research import prepared
from test_inquiry_revision import write


def trajectory(*controls, channels=()):
    return {"kind": "trajectory", "sequence": [{"u": value} for value in controls],
            "channels": list(channels)}


def pair(seed):
    return [TemporalInquiryLab(task_index=1, apparatus_seed=seed, instrument_mode=mode)
            for mode in ("finite2", "recurrent")]


@pytest.mark.parametrize("seed", [113, 823])
def test_short_reset_evidence_is_identical_but_long_evidence_separates(seed):
    finite, recurrent = pair(seed)
    assert finite.public_spec() == recurrent.public_spec()
    assert finite.task_interface == recurrent.task_interface
    # Include every short word on three controls, not merely a zero prefix.
    for length in range(1, 4):
        for controls in product((-.8, 0., .7), repeat=length):
            records = [lab.execute(trajectory(*controls, channels=("meter",)))
                       for lab in (finite, recurrent)]
            for key in ("rows", "measurements"):
                left, right = records[0][key], records[1][key]
                values = (lambda items: [item["target"] for item in items]) if key == "rows" else (
                    lambda items: [item["values"]["meter"] for item in items])
                assert values(left) == pytest.approx(values(right), abs=1e-12)
    long_records = [lab.execute(trajectory(.8, 0., 0., 0., 0., 0., channels=("meter",)))
                    for lab in (finite, recurrent)]
    assert abs(long_records[0]["rows"][3]["target"]
               - long_records[1]["rows"][3]["target"]) > 1e-5
    assert abs(long_records[0]["measurements"][3]["values"]["meter"]
               - long_records[1]["measurements"][3]["values"]["meter"]) > 1e-5


@pytest.mark.parametrize("mode", ["finite2", "recurrent"])
def test_prospective_inputs_match_purchase_and_do_not_read_physics(mode, monkeypatch):
    lab = TemporalInquiryLab(task_index=1, instrument_mode=mode)
    action = trajectory(.8, -.3, 0., .1, channels=("meter",))
    supplied = deepcopy(action)
    before = lab.costs

    def unavailable(*args, **kwargs):
        raise AssertionError("prospective inputs must not acquire physical evidence")

    with monkeypatch.context() as context:
        context.setattr(lab._laboratory, "acquire", unavailable)
        inputs = lab.prediction_rows(action)
    assert lab.costs == before and action == supplied
    assert all(row.target == 0. for row in inputs)
    measured = lab.observation_rows(lab.execute(action))
    assert [(row.features, row.history) for row in inputs] == [
        (row.features, row.history) for row in measured]
    assert all(set(row.features) == {"u"} for row in inputs)
    assert all(set(point) == {"u"} for row in inputs for point in row.history)
    assert lab.prediction_rows(action)[0].history == ()


@pytest.mark.parametrize("arm", ["code_react", "hegel"])
def test_common_archive_comparison_uses_the_real_temporal_public_inputs(tmp_path, arm):
    agent, lab = prepared(tmp_path, arm), TemporalInquiryLab(task_index=1)
    agent._execute("query", {"actions": [trajectory(0.)]}, lab)
    observed = agent.rows[0].target
    write(agent, "old.py", f"def predict(row): return {observed!r}\n", lab)
    write(agent, "new.py", f"def predict(row): return {observed!r} + len(row['history'])\n", lab)
    before = lab.costs
    result = agent._execute("run", {"operator": "compare_experiments",
        "models": ["old.py", "new.py"], "experiments": [
            {"id": "reset", "actions": [trajectory(0.)]},
            {"id": "history", "actions": [trajectory(0., 0.)]}]}, lab)["comparison"]
    assert result["eligible_models"] == ["old.py", "new.py"]
    assert result["selected_experiment"]["id"] == "history"
    assert lab.costs == before and len(agent.rows) == 1
    queried = agent._execute("query", result["query_arguments"], lab)
    assert queried["revision"]["forecasts"][0]["prospective_mse"] == pytest.approx(0.)
    assert queried["revision"]["forecasts"][1]["prospective_mse"] == pytest.approx(.5)


@pytest.mark.parametrize("arm", ["code_react", "hegel"])
def test_paid_refutation_stops_fresh_purchase_in_fixed_apparatus(tmp_path, arm):
    agent, lab = prepared(tmp_path, arm), TemporalInquiryLab(task_index=1)
    agent._execute("query", {"actions": [trajectory(.8, -.4)]}, lab)
    write(agent, "wrong.py", "def predict(row): return 1000.\n", lab)
    before = lab.costs
    result = agent._execute("finish", {"model": "wrong.py"}, lab)
    assert "paid evidence contradicts" in result["error"]
    assert "fresh_confirmation" not in result
    assert lab.costs == before


def test_old_calibration_evidence_does_not_block_settled_confirmation():
    lab = TemporalInquiryLab(task_index=1,
        calibration_change={"after_rows": 3, "apparatus_seed": 823})
    assert lab.confirmation_precondition({"mse": 1.}, mse_threshold=1e-10) is None
    lab.execute(trajectory(.8, -.3, .5))
    lab.execute(trajectory(.2))  # The update has now occurred.
    assert lab.confirmation_precondition({"mse": 1.}, mse_threshold=1e-10) is None
    fixed = TemporalInquiryLab(task_index=1)
    assert fixed.confirmation_precondition({"mse": 0.}, mse_threshold=1e-10) is None
