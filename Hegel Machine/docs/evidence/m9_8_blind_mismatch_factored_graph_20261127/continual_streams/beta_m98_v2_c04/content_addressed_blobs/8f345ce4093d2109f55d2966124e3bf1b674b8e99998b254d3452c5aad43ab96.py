"""Robust forward calibration and inversion for the four-channel rosette."""
from itertools import combinations

import numpy as np

_PAIRS = tuple(combinations(range(4), 2))


def _features(x):
    cross = [x[:, i] * x[:, j] for i, j in _PAIRS]
    return np.column_stack((np.ones(len(x)), x, x*x, *cross, x**3))


def _fit(ref, raw):
    a = _features(ref)
    w = np.ones(len(ref))
    for _ in range(30):
        sw = np.sqrt(w)[:, None]
        beta = np.linalg.lstsq(a * sw, raw * sw, rcond=None)[0]
        residual = np.linalg.norm(a @ beta - raw, axis=1)
        scale = max(np.median(residual), 1e-10)
        w = 1 / (1 + (residual / (2 * scale))**2)
    clean = residual < 8 * scale
    if np.linalg.matrix_rank(a[clean]) == a.shape[1]:
        beta = np.linalg.lstsq(a[clean], raw[clean], rcond=None)[0]
    return beta


def _jacobian(x, beta):
    # Axes are (sample, raw output, reference input).
    j = np.broadcast_to(beta[1:5].T, (len(x), 4, 4)).copy()
    j += (2*x)[:, None, :] * beta[5:9].T
    for k, (i, l) in enumerate(_PAIRS, 9):
        j[:, :, i] += x[:, l, None] * beta[k]
        j[:, :, l] += x[:, i, None] * beta[k]
    j += (3*x*x)[:, None, :] * beta[15:19].T
    return j


def _invert(raw, beta):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    x = np.linalg.solve(beta[1:5].T, (y - beta[0]).T).T
    for _ in range(30):
        r = _features(x) @ beta - y
        err = np.linalg.norm(r, axis=1)
        active = err > 1e-12 * (1 + np.linalg.norm(y, axis=1))
        if not np.any(active):
            break
        current = x[active]
        step = np.linalg.solve(_jacobian(current, beta), r[active, :, None])[..., 0]
        t = np.ones(len(current))
        target = y[active]
        for _ in range(16):
            trial = current - t[:, None] * step
            improved = np.linalg.norm(_features(trial) @ beta - target, axis=1) < err[active]
            if np.all(improved):
                break
            t[~improved] *= 0.5
        x[active] = np.where(improved[:, None], trial, current)
        if not np.any(improved):
            break
    return x.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a copy of the study with observations and sweeps in reference units."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    beta = _fit(ref, np.asarray(sweeps['raw_rosette'], dtype=float))
    single = sweeps['single_bank_sweeps']
    calibrated = dict(study, y_obs_raw=_invert(study['y_obs_raw'], beta))
    calibrated['sensor_calibration_sweeps'] = dict(
        sweeps,
        raw_rosette=ref.copy(),
        single_bank_sweeps=dict(single, raw_readout=np.asarray(single['ref_signal']).copy()),
    )
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        calibrated['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], beta)) for pulse in pulses
        ]
    return calibrated
