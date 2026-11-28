"""Calibrated cosine panel operator with separate scalar and trajectory readouts."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import minimize_scalar


def _fit(design, values):
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _cosine(x, frequency):
    return np.column_stack((np.ones(len(x)), np.cos(frequency * x) - 1.0))


def _panel_operator(panel, frequency_hint):
    u = np.asarray(panel['u'], dtype=float)
    v = np.asarray(panel['v'], dtype=float)

    def residual(frequency):
        a = _cosine(u, frequency)
        err = a @ _fit(a, v) - v
        return float(np.sum(err**2))

    # Profile out the mixing matrix. Bracket the lowest sampled residual
    # before refining so a distant periodic minimum is not selected blindly.
    grid = frequency_hint * np.linspace(0.55, 1.85, 27)
    j = int(np.argmin([residual(k) for k in grid]))
    lo, hi = grid[max(0, j - 1)], grid[min(j + 1, len(grid) - 1)]
    result = minimize_scalar(residual, bounds=(lo, hi), method='bounded',
                             options={'xatol': 1e-8})
    frequency = min((grid[j], result.x), key=residual)
    return frequency, _fit(_cosine(u, frequency), v)


def _readout(b):
    products = [b[:, i] * b[:, j]
                for i, j in combinations_with_replacement(range(b.shape[1]), 2)]
    return np.column_stack([np.ones(len(b)), b, *products])


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    calibration = inputs['node_00']
    gain = np.asarray(calibration['gain'])
    offset = np.asarray(calibration['offset'])
    train = study['train']
    # The frozen coordinate calibration is reusable; the frozen odd panel-B
    # basis cannot represent the measured even response and is refitted here.
    hint = max(abs(float(study['telemetry_vec'][-1])), 0.1)
    frequency, mixing = _panel_operator(study['panel_b'], hint)

    def design(x):
        q = (np.asarray(x, dtype=float) - offset) / gain
        return _readout(_cosine(q, frequency) @ mixing)

    a = design(train['x'])
    new = design(study['test']['x'])
    y = new @ _fit(a, np.asarray(train['y']))
    traj = new @ _fit(a, np.asarray(train['traj']))
    last = traj[:, -1]
    diag = np.array([y.mean(), y.std(), last.mean(), last.std()])
    return {'y_hat': y, 'state_pred': traj, 'diagnostics_vec': diag}
