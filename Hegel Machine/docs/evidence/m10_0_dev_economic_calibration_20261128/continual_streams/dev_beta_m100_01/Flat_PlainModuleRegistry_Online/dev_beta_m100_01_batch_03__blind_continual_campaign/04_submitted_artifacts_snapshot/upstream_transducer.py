"""Robust calibration of the coupled four-channel rosette."""
from itertools import combinations

import numpy as np


_PAIRS = tuple(combinations(range(4), 2))


def _basis(x):
    return np.column_stack((np.ones(len(x)), x,
                            *(x[:, i] * x[:, j] for i, j in _PAIRS), x**3))


def _fit(ref, raw):
    a = _basis(ref)
    coef = np.linalg.lstsq(a, raw, rcond=None)[0]
    # Fit whole rosettes with Huber weights, then refit without EMI rows.
    for _ in range(24):
        r = np.linalg.norm(raw - a @ coef, axis=1)
        cutoff = max(1e-10, 1.5 * np.median(r))
        w = np.sqrt(cutoff / np.maximum(r, cutoff))
        coef = np.linalg.lstsq(a * w[:, None], raw * w[:, None], rcond=None)[0]
    r = np.linalg.norm(raw - a @ coef, axis=1)
    clean = r <= max(8 * np.median(r), 1e-8)
    if np.any(~clean) and np.linalg.matrix_rank(a[clean]) == a.shape[1]:
        coef = np.linalg.lstsq(a[clean], raw[clean], rcond=None)[0]
    return coef


def _invert(raw, coef):
    values = np.asarray(raw, dtype=float)
    target = values.reshape(-1, 4)
    origin, linear = coef[0], coef[1:5]
    x = np.linalg.solve(linear.T, (target - origin).T).T.copy()
    tol = 1e-11 * (1 + np.max(np.abs(target), axis=1))
    for _ in range(40):
        r = _basis(x) @ coef - target
        err = np.max(np.abs(r), axis=1)
        idx = np.flatnonzero(err > tol)
        if not idx.size:
            return x.reshape(values.shape)
        z = x[idx]
        # Transposed Jacobian: input axis first, output axis second.
        jac_t = linear + 3 * z[:, :, None]**2 * coef[None, 11:15, :]
        for k, (i, j) in enumerate(_PAIRS, start=5):
            jac_t[:, i] += z[:, j, None] * coef[k]
            jac_t[:, j] += z[:, i, None] * coef[k]
        step = np.linalg.solve(jac_t.transpose(0, 2, 1), r[idx, :, None])[..., 0]
        # Near the solution take the full Newton step; otherwise backtrack.
        for power in range(20):
            trial = x[idx] - (0.5**power) * step
            better = np.max(np.abs(_basis(trial) @ coef - target[idx]), axis=1) < err[idx]
            x[idx[better]] = trial[better]
            if better.all():
                break
            idx, step = idx[~better], step[~better]
        else:
            break
    raise RuntimeError('Rosette calibration inverse did not converge')


def calibrate_upstream(study: dict) -> dict:
    """Return a study in canonical coordinates without changing its input."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
    # The canonical interface itself also accepts already-normalized sweeps.
    inverse = (lambda v: np.asarray(v).copy()) if np.array_equal(ref, raw) else None
    if inverse is None:
        coef = _fit(ref, raw)

        def inverse(v):
            return _invert(v, coef)

    out = study.copy()
    out['y_obs_raw'] = inverse(study['y_obs_raw'])
    out_sweeps = sweeps.copy()
    out_sweeps['raw_rosette'] = np.asarray(sweeps['ref_rosette']).copy()
    single = sweeps['single_bank_sweeps']
    out_sweeps['single_bank_sweeps'] = {
        **single, 'raw_readout': np.asarray(single['ref_signal']).copy(),
    }
    out['sensor_calibration_sweeps'] = out_sweeps
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        out['reference_tracer_pulses'] = [
            {**pulse, 'y_pulse_raw': inverse(pulse['y_pulse_raw'])}
            if pulse['y_pulse_raw'] is not None else pulse.copy()
            for pulse in pulses
        ]
    return out
