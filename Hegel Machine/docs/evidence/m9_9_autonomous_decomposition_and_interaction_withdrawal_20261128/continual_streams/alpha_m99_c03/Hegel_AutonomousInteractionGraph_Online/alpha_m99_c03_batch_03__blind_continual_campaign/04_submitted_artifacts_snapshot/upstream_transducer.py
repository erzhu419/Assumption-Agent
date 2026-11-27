"""Robust forward calibration and inversion of the four load-cell bridges."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares


_LINEAR = [()] + [(i,) for i in range(4)]
_FULL = _LINEAR + [t for degree in (2, 3)
                    for t in combinations_with_replacement(range(4), degree)]
_LOCAL = [tuple(sorted((i, (i + 1) % 4))) for i in range(4)]
_LOCAL += [(i, i, i) for i in range(4)]
_RING = _LINEAR + _LOCAL


def _features(x, terms):
    return np.column_stack([np.prod(x[:, t], axis=1) for t in terms])


def _regress(a, y, mask, weight):
    b = np.zeros(mask.shape)
    for j in range(y.shape[1]):
        b[mask[:, j], j] = np.linalg.lstsq(
            a[:, mask[:, j]] * weight[:, None], y[:, j] * weight, rcond=None)[0]
    return b


def _robust(a, y, mask, initial=None):
    # The spikes affect bridge outputs jointly, so weights are shared by row.
    if initial is None:
        initial = np.ones(len(y), dtype=bool)
    b = _regress(a, y, mask, initial.astype(float))
    for _ in range(25):
        r = np.linalg.norm(y - a @ b, axis=1)
        s = max(1e-12, np.median(r[initial]))
        new = _regress(a, y, mask, initial / np.hypot(1.0, r / (2.0 * s)))
        if np.max(np.abs(new - b)) < 1e-11:
            b = new
            break
        b = new
    return b


def _fit_forward(x, y):
    # Start by excluding only gross excursions relative to the calibration range.
    center = np.median(y, axis=0)
    magnitude = np.linalg.norm(y - center, axis=1)
    initial = magnitude <= 6 * max(np.median(magnitude), 1e-12)
    a = _features(x, _RING)
    mask = np.vstack((np.ones((5, 4), dtype=bool), np.eye(4, dtype=bool),
                      np.eye(4, dtype=bool)))
    b = _robust(a, y, mask, initial)
    error = np.linalg.norm(y - a @ b, axis=1)
    # The bridge law includes linear mixing, a self-cubic, and a local pair
    # product. Use more general cubic coupling if that law does not fit.
    if np.median(error[initial]) <= 0.001 * max(np.median(magnitude), 1e-12):
        return _RING, b
    a = _features(x, _FULL)
    if initial.sum() <= a.shape[1]:
        return _RING, b
    b = _robust(a, y, np.ones((a.shape[1], 4), dtype=bool), initial)
    return _FULL, b


def _jacobian(x, terms, b):
    # Result axes are row, measured bridge, true load.
    jac = np.empty((len(x), 4, 4))
    for i in range(4):
        subset = [k for k, t in enumerate(terms) if i in t]
        reduced = []
        for k in subset:
            t = list(terms[k])
            t.remove(i)
            reduced.append(tuple(t))
        jac[:, :, i] = _features(x, reduced) @ (
            b[subset] * np.array([terms[k].count(i) for k in subset])[:, None])
    return jac


def _invert(raw, terms, b):
    shape = raw.shape
    y = np.asarray(raw, dtype=float).reshape(-1, 4)
    x = (y - b[0]) @ np.linalg.pinv(b[1:5])
    for _ in range(25):
        r = _features(x, terms) @ b - y
        active = np.flatnonzero(np.max(np.abs(r), axis=1) > 1e-11)
        if not len(active):
            break
        xa, ra = x[active], r[active]
        jac = _jacobian(xa, terms, b)
        delta = np.einsum('nji,ni->nj', np.linalg.pinv(jac), ra)
        target = np.sum(ra**2, axis=1)
        step = np.ones(len(active))
        for _ in range(12):
            trial = xa - step[:, None] * delta
            improved = np.sum((_features(trial, terms) @ b - y[active])**2,
                              axis=1) < target
            if np.all(improved):
                break
            step[~improved] *= 0.5
        x[active[improved]] = trial[improved]
        if not np.any(improved):
            break
    remaining = np.flatnonzero(np.max(np.abs(_features(x, terms) @ b - y), axis=1) > 1e-8)
    for k in remaining:
        x[k] = least_squares(lambda v: (_features(v[None, :], terms) @ b)[0] - y[k],
                             x[k], jac=lambda v: _jacobian(v[None, :], terms, b)[0]).x
    return x.reshape(shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return canonical load coordinates without modifying the input dataset."""
    calibration = dataset['calibration_load_cells']
    true = np.asarray(calibration['true_channel_loads'], dtype=float)
    raw = np.asarray(calibration['bridge_raw'], dtype=float)
    result = dict(dataset)
    load_fields = ('past_channel_raw', 'hold_channel_raw')
    if not np.array_equal(true, raw):
        terms, coefficients = _fit_forward(true, raw)
        for name in ('train', 'test', 'chamber_verification_trials'):
            block = dataset.get(name)
            if block is not None:
                result[name] = {**block, **{key: _invert(block[key], terms, coefficients)
                                            for key in load_fields}}
    sweeps = calibration['single_channel_sweeps']
    result['calibration_load_cells'] = {
        **calibration, 'bridge_raw': calibration['true_channel_loads'].copy(),
        'single_channel_sweeps': {**sweeps, 'cmd_stress': sweeps['true_stress'].copy()},
    }
    return result
