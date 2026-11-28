"""Robust forward calibration of the four-channel rosette and its inverse."""
from itertools import combinations

import numpy as np


_I, _J = np.array(list(combinations(range(4), 2))).T


def _design(x):
    return np.column_stack((np.ones(len(x)), x, x[:, _I] * x[:, _J], x**3))


def _fit(ref, raw):
    a = _design(ref)
    coef = np.linalg.lstsq(a, raw, rcond=None)[0]
    # Downweight a whole sweep row when any of its channels has an EMI spike.
    for _ in range(30):
        error = np.linalg.norm(raw - a @ coef, axis=1)
        scale = max(np.median(error), 1e-12)
        w = 1 / np.hypot(1, error / scale)
        update = np.linalg.lstsq(w[:, None] * a, w[:, None] * raw, rcond=None)[0]
        if np.max(np.abs(update - coef)) < 1e-11:
            coef = update
            break
        coef = update
    # Recover ordinary least-squares efficiency on the clean rows.
    error = np.linalg.norm(raw - a @ coef, axis=1)
    keep = error <= 5 * max(np.median(error), 1e-12)
    if np.count_nonzero(keep) >= a.shape[1]:
        refined, _, rank, _ = np.linalg.lstsq(a[keep], raw[keep], rcond=None)
        if rank == a.shape[1]:
            coef = refined
    return coef


def _inverse(raw, coef):
    array = np.asarray(raw, dtype=float)
    target = array.reshape(-1, 4)
    linear = coef[1:5]
    x = np.linalg.solve(linear.T, (target - coef[0]).T).T
    for _ in range(35):
        residual = _design(x) @ coef - target
        norm = np.linalg.norm(residual, axis=1)
        active = np.flatnonzero(norm > 1e-12 * (1 + np.linalg.norm(target, axis=1)))
        if not len(active):
            break
        xa = x[active]
        # J has reference-input rows and raw-output columns, like linear.
        jac = linear[None, :, :] + 3 * xa[:, :, None]**2 * coef[None, 11:15, :]
        for i, j, c in zip(_I, _J, coef[5:11]):
            jac[:, i, :] += xa[:, j, None] * c
            jac[:, j, :] += xa[:, i, None] * c
        step = np.linalg.solve(jac.transpose(0, 2, 1), residual[active, :, None])[..., 0]
        # Backtrack independently for each reading, keeping local branch continuity.
        for _ in range(16):
            trial = x[active] - step
            better = np.linalg.norm(_design(trial) @ coef - target[active], axis=1) < norm[active]
            x[active[better]] = trial[better]
            if np.all(better):
                break
            active, step = active[~better], 0.5 * step[~better]
        else:
            if not np.any(better):
                break
    return x.reshape(array.shape)


def calibrate_upstream(study: dict) -> dict:
    """Put observations, tracer pulses, and calibration sweeps in reference coordinates."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    coef = _fit(ref, np.asarray(sweeps['raw_rosette'], dtype=float))
    out = dict(study, y_obs_raw=_inverse(study['y_obs_raw'], coef))
    single = sweeps['single_bank_sweeps']
    out['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=ref.copy(),
        single_bank_sweeps=dict(single, raw_readout=np.asarray(single['ref_signal']).copy()),
    )
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        out['reference_tracer_pulses'] = [
            pulse if pulse is None or pulse.get('y_pulse_raw') is None
            else dict(pulse, y_pulse_raw=_inverse(pulse['y_pulse_raw'], coef))
            for pulse in pulses
        ]
    return out
