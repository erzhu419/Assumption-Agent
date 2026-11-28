"""Robust four-channel bridge calibration in the forward measurement coordinates."""
from itertools import combinations_with_replacement

import numpy as np


_PAIRS = np.array(list(combinations_with_replacement(range(4), 2)))
_I, _J = _PAIRS.T
_EYE = np.eye(4)
_CYCLIC = [np.r_[np.arange(5), 5 + np.flatnonzero(
    np.all(_PAIRS == sorted((j, (j + 1) % 4)), axis=1))[0], 15 + j]
    for j in range(4)]


def _basis(x):
    return np.column_stack((np.ones(len(x)), x, x[:, _I] * x[:, _J], x**3))


def _robust_fit(a, y):
    """Downweight entire calibration records with inconsistent bridge readings."""
    b = np.linalg.lstsq(a, y, rcond=None)[0]
    for _ in range(45):
        r = np.linalg.norm(a @ b - y, axis=1)
        s = max(2e-7, 1.6 * np.median(r))
        w = 1 / np.hypot(1., r / s)
        updated = np.linalg.lstsq(a * w[:, None], y * w[:, None], rcond=None)[0]
        if np.max(np.abs(updated - b)) < 1e-10:
            return updated
        b = updated
    return b


def _cyclic_seed(a, y):
    """Try the compact bridge first so spikes cannot hide in extra terms."""
    b = np.zeros((a.shape[1], y.shape[1]))
    for j, cols in enumerate(_CYCLIC):
        b[cols, j] = _robust_fit(a[:, cols], y[:, j:j+1])[:, 0]
    return b


def _inliers(radius):
    return radius <= 6 * max(np.median(radius), 2e-7)


def _fit_bridge(x, y):
    a = _basis(x)
    seed = _cyclic_seed(a, y)
    radius = np.linalg.norm(a @ seed - y, axis=1)
    signal = np.median(np.linalg.norm(y - np.median(y, axis=0), axis=1))
    # Only trust the compact model's spike mask if it already fits at noise scale.
    if np.median(radius) <= 0.002 * signal:
        keep = _inliers(radius)
        design = a[keep]
        if len(design) < 2 * a.shape[1] or np.linalg.matrix_rank(design) < a.shape[1]:
            for j, cols in enumerate(_CYCLIC):
                seed[cols, j] = np.linalg.lstsq(design[:, cols], y[keep, j], rcond=None)[0]
            return seed
        b = np.linalg.lstsq(design, y[keep], rcond=None)[0]
    else:
        b = _robust_fit(a, y)
        keep = _inliers(np.linalg.norm(a @ b - y, axis=1))
        design = a[keep]
        if np.linalg.matrix_rank(design) < a.shape[1]:
            return b

    # Retain interactions that exceed the resolution of the inlier design.
    resid = y[keep] - design @ b
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid, axis=0)), axis=0)
    leverage = np.linalg.norm(np.linalg.pinv(design), axis=1)
    active = np.abs(b) > 5 * leverage[:, None] * np.maximum(sigma, 2e-7)
    active[:5] = True
    compact = np.zeros_like(b)
    for j in range(4):
        cols = active[:, j]
        compact[cols, j] = np.linalg.lstsq(design[:, cols], y[keep, j], rcond=None)[0]
    return compact


def _jacobian(x, b):
    quadratics = x[:, _J, None] * _EYE[_I] + x[:, _I, None] * _EYE[_J]
    return (b[1:5].T[None, :, :] + b[5:15].T @ quadratics
            + b[15:].T[None, :, :] * (3 * x[:, None, :] ** 2))


def _invert(raw, b):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    x = np.linalg.solve(b[1:5].T, (y - b[0]).T).T.copy()
    for _ in range(35):
        r = _basis(x) @ b - y
        err = np.max(np.abs(r), axis=1)
        live = err > 2e-12 * (1 + np.max(np.abs(y), axis=1))
        if not np.any(live):
            break
        z, target, old = x[live], y[live], err[live]
        step = np.linalg.solve(_jacobian(z, b), r[live, :, None])[..., 0]
        scale = np.ones((len(z), 1))
        trial = z - step
        for _ in range(24):
            worse = np.max(np.abs(_basis(trial) @ b - target), axis=1) >= old
            if not np.any(worse):
                break
            scale[worse] *= 0.5
            trial[worse] = z[worse] - scale[worse] * step[worse]
        x[live] = trial
    return x.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return a canonical-coordinate dataset, leaving the input unchanged."""
    cal = dataset['calibration_load_cells']
    loads = np.asarray(cal['true_channel_loads'], dtype=float)
    b = _fit_bridge(loads, np.asarray(cal['bridge_raw'], dtype=float))

    def convert(split):
        return dict(split, **{key: _invert(split[key], b)
                             for key in ('past_channel_raw', 'hold_channel_raw') if key in split})

    sweeps = cal['single_channel_sweeps']
    result = dict(dataset, calibration_load_cells=dict(
        cal, bridge_raw=cal['true_channel_loads'].copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=sweeps['true_stress'].copy())),
        train=convert(dataset['train']), test=convert(dataset['test']))
    trials = dataset.get('chamber_verification_trials')
    if isinstance(trials, dict):
        result['chamber_verification_trials'] = convert(trials)
    elif isinstance(trials, (list, tuple)):
        result['chamber_verification_trials'] = type(trials)(convert(trial) for trial in trials)
    return result
