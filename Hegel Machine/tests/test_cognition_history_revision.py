"""History revisions retain interactions and select on whole unseen trajectories."""

from dataclasses import asdict

import numpy as np
import pytest

from hegel_machine.cognition.history_revision import revise_history
from hegel_machine.cognition.models import Observation


def trajectories(lengths, response, seed=41):
    rng = np.random.default_rng(seed)
    rows = []
    for group, length in enumerate(lengths):
        controls = rng.uniform(-1, 1, length)
        for index, u in enumerate(controls):
            history = tuple({"u": float(x)} for x in controls[:index])
            coordinates = [float(u), *reversed(controls[:index]), 0.0, 0.0, 0.0]
            rows.append(Observation({"u": float(u)}, response(coordinates, group), history))
    return rows


def test_new_coordinate_keeps_old_and_new_cross_lag_interactions():
    def response(x, group):
        return 0.3 + 0.7 * x[0] ** 2 - 0.4 * x[1] * x[2] + 1.2 * x[2] * x[3]

    rows = trajectories([24] * 9, response)
    proposal = revise_history([asdict(row) for row in rows])
    result = proposal.resultdict

    assert result["status"] == "revised" and proposal.selected_lags == 3
    assert result["base"]["heldout_mse"] > 0.02
    assert result["revision"]["heldout_mse"] < 1e-24
    assert result["revision"]["training_rank"] == result["revision"]["feature_count"] == 15
    assert result["selected_refit"]["train_mse"] < 1e-24
    unseen = trajectories([24] * 2, response, seed=83)
    assert [proposal.predict(row) for row in unseen] == pytest.approx([row.target for row in unseen], abs=1e-10)


def test_holdout_keeps_whole_trajectories_and_rejects_worse_extension():
    # The extra interaction is present only in training trajectories. Fitting
    # all rows before selection would expose the held-out trajectories' labels.
    def response(x, group):
        return x[0] + (x[2] * x[3] if group < 4 else 0.0)

    rows = trajectories([20, 25, 22, 29, 27, 31], response)
    proposal = revise_history(rows)
    result = proposal.resultdict

    assert result["split"] == {"train_trajectories": [0, 1, 2, 3],
                                "holdout_trajectories": [4, 5],
                                "train_rows": 96, "holdout_rows": 58, "holdout_start_row": 96}
    assert not rows[result["split"]["holdout_start_row"]].history
    assert result["revision"]["train_mse"] < 1e-24
    assert result["revision"]["heldout_mse"] > result["base"]["heldout_mse"]
    assert result["status"] == "base_retained" and proposal.selected_lags == 2


def test_short_trajectory_holdout_cannot_support_an_extra_history_coordinate():
    rows = trajectories([24] * 6 + [3] * 3, lambda x, group: x[0] + x[2] * x[3])
    result = revise_history(rows).resultdict
    assert result["revision"]["training_rank"] == 15
    assert result["revision"]["holdout_rank"] < 15
    assert result["status"] == "unsupported" and result["selected_lags"] == 2


def test_export_preserves_prediction_and_fit_updates_the_actual_coefficients():
    rows = trajectories([24] * 6, lambda x, group: x[0] ** 2 + x[2] * x[3])
    proposal = revise_history(rows)
    namespace = {}
    exec(proposal.to_code(), namespace)

    expected = [proposal.predict(row) for row in rows]
    assert [namespace["predict"](asdict(row)) for row in rows] == pytest.approx(expected, abs=1e-12)
    shifted = [Observation(row.features, row.target + 2.0, row.history) for row in rows]
    namespace["fit"]([asdict(row) for row in shifted])
    assert [namespace["predict"](row) for row in shifted] == pytest.approx(
        [value + 2.0 for value in expected], abs=1e-10)
