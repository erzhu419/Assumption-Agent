"""Fit and invert the study's coupled four-channel rosette response."""
from itertools import combinations_with_replacement

import numpy as np


_TERMS = tuple(t for degree in range(4)
               for t in combinations_with_replacement(range(4), degree))


def _basis(x):
    return np.column_stack([np.prod(x[:, t], axis=1) if t else np.ones(len(x))
                            for t in _TERMS])


def _fit(ref, raw):
    a = _basis(ref)
    coef = np.linalg.lstsq(a, raw, rcond=None)[0]
    # Weight entire rows: an EMI burst can corrupt multiple channels together.
    floor = 1e-12 * max(1., np.median(np.abs(raw)))
    for _ in range(35):
        error = np.linalg.norm(a @ coef - raw, axis=1)
        cutoff = max(3 * np.median(error), floor)
        w = cutoff / np.maximum(cutoff, error)
        new = np.linalg.lstsq(a * w[:, None], raw * w[:, None], rcond=None)[0]
        converged = np.max(np.abs(new - coef)) <= floor
        coef = new
        if converged:
            break

    error = np.linalg.norm(a @ coef - raw, axis=1)
    clean = error <= max(6 * np.median(error), floor)
    ac, yc = a[clean], raw[clean]
    coef, _, rank, _ = np.linalg.lstsq(ac, yc, rcond=None)
    if rank == ac.shape[1] and len(ac) > rank:
        # Discover support on the current study, not from public coefficients.
        noise = np.linalg.norm(yc - ac @ coef, axis=0) / np.sqrt(len(ac) - rank)
        se = np.linalg.norm(np.linalg.pinv(ac), axis=1)[:, None] * noise
        keep = np.abs(coef) > np.maximum(6 * se, floor)
        keep[:5] = True
        for j in range(4):
            selected = keep[:, j]
            coef[~selected, j] = 0.
            coef[selected, j] = np.linalg.lstsq(ac[:, selected], yc[:, j], rcond=None)[0]
    return coef


def _invert(raw, coef):
    shape = np.shape(raw)
    y = np.asarray(raw, dtype=float).reshape(-1, 4)
    if not len(y):
        return y.reshape(shape)
    x = np.linalg.solve(coef[1:5].T, (y - coef[0]).T).T
    # Compact forward map and analytic Jacobian for damped Newton iteration.
    active = [(t, c) for t, c in zip(_TERMS, coef) if t and np.any(c)]

    def residual(z):
        r = coef[0] - y
        for t, c in active:
            r += np.prod(z[:, t], axis=1)[:, None] * c
        return r

    r = residual(x)
    tol = 1e-12 * (1 + np.linalg.norm(y, axis=1))
    for _ in range(40):
        err = np.linalg.norm(r, axis=1)
        if np.all(err <= tol):
            break
        jac = np.zeros((len(y), 4, 4))
        for t, c in active:
            for k, j in enumerate(t):
                d = np.prod(x[:, t[:k] + t[k+1:]], axis=1)
                jac[:, :, j] += d[:, None] * c
        delta = np.linalg.solve(jac, r[..., None])[..., 0]
        step = np.ones(len(y))
        pending = err > tol
        for _ in range(24):
            trial = x - (step * pending)[:, None] * delta
            trial_r = residual(trial)
            accept = pending & (np.linalg.norm(trial_r, axis=1) < err)
            x[accept], r[accept] = trial[accept], trial_r[accept]
            pending &= ~accept
            if not np.any(pending):
                break
            step[pending] *= .5
    return x.reshape(shape)


def calibrate_upstream(study: dict) -> dict:
    """Return the study in reference coordinates without modifying its inputs."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    coef = _fit(ref, np.asarray(sweeps['raw_rosette'], dtype=float))
    bank = sweeps['single_bank_sweeps']
    result = dict(study, y_obs_raw=_invert(study['y_obs_raw'], coef))
    result['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=sweeps['ref_rosette'].copy(),
        single_bank_sweeps=dict(bank, raw_readout=bank['ref_signal'].copy()))
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        result['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coef))
            if pulse is not None and pulse.get('y_pulse_raw') is not None else pulse
            for pulse in pulses]
    return result
