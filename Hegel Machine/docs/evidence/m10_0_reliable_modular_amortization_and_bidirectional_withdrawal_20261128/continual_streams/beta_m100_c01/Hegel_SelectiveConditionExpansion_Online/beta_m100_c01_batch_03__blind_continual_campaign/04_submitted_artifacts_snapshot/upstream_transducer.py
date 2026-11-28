"""Fit the rosette in forward coordinates, then invert the fitted response."""

from itertools import combinations_with_replacement

import numpy as np


_TERMS = tuple(t for d in range(1, 4)
               for t in combinations_with_replacement(range(4), d))


def _basis(x):
    return np.column_stack([np.ones(len(x))] +
                           [x[:, t].prod(axis=1) for t in _TERMS])


def _fit(ref, raw):
    design = _basis(ref)
    # An EMI spike contaminates an acquisition row, not four independent fits.
    weights = np.ones(len(ref))
    for _ in range(25):
        coef = np.linalg.lstsq(design * weights[:, None],
                              raw * weights[:, None], rcond=None)[0]
        error = np.linalg.norm(raw - design @ coef, axis=1)
        scale = max(np.median(error), 1e-12)
        weights = 1 / np.hypot(1, error / (2 * scale))
    clean = error <= 8 * scale
    if not np.all(clean):
        trimmed, _, rank, _ = np.linalg.lstsq(design[clean], raw[clean], rcond=None)
        if rank == design.shape[1]:
            coef = trimmed
    return coef


def _jacobian(x, coef):
    jac = np.zeros((len(x), 4, 4))
    for term, c in zip(_TERMS, coef[1:]):
        for i, j in enumerate(term):
            product = x[:, term[:i] + term[i + 1:]].prod(axis=1)
            jac[:, :, j] += product[:, None] * c
    return jac


def _invert(raw, coef):
    array = np.asarray(raw, dtype=float)
    y = array.reshape(-1, 4)
    x = np.linalg.solve(coef[1:5].T, (y - coef[0]).T).T.copy()
    for _ in range(32):
        residual = _basis(x) @ coef - y
        error = np.linalg.norm(residual, axis=1)
        active = error > 1e-12 * (1 + np.linalg.norm(y, axis=1))
        if not np.any(active):
            break
        old = x[active]
        target = y[active]
        step = np.linalg.solve(_jacobian(old, coef), residual[active, :, None])[..., 0]
        factor = np.ones(len(old))
        for _ in range(16):
            trial = old - factor[:, None] * step
            improves = np.linalg.norm(_basis(trial) @ coef - target, axis=1) < error[active]
            if np.all(improves):
                break
            factor[~improves] *= 0.5
        x[active] = np.where(improves[:, None], trial, old)
    return x.reshape(array.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a calibrated study without modifying the supplied study or sweeps."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
    coef = _fit(ref, raw)

    out = dict(study)
    out['y_obs_raw'] = _invert(study['y_obs_raw'], coef)
    bank = sweeps['single_bank_sweeps']
    out['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=sweeps['ref_rosette'].copy(),
        single_bank_sweeps=dict(bank, raw_readout=bank['ref_signal'].copy()))
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        out['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coef))
            if pulse is not None and pulse.get('y_pulse_raw') is not None else pulse
            for pulse in pulses
        ]
    return out
