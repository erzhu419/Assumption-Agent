"""Fit and invert the four-channel rosette response in each study."""
from itertools import combinations_with_replacement

import numpy as np


_TERMS = [()] + [term for degree in (1, 2, 3)
                 for term in combinations_with_replacement(range(4), degree)]
_TERM_ID = {term: i for i, term in enumerate(_TERMS)}
_COLUMNS = np.array([term + (4,) * (3 - len(term)) for term in _TERMS])
_DERIVATIVE = np.zeros((4, len(_TERMS), len(_TERMS)))
for i, term in enumerate(_TERMS):
    for j in set(term):
        k = term.index(j)
        _DERIVATIVE[j, _TERM_ID[term[:k] + term[k + 1:]], i] = term.count(j)
_CYCLIC = [list(range(5)) + [_TERM_ID[tuple(sorted((j, (j + 1) % 4)))],
                             _TERM_ID[(j, j, j)]] for j in range(4)]


def _design(x):
    padded = np.column_stack((x, np.ones(len(x))))
    return padded[:, _COLUMNS].prod(axis=2)


def _fit(reference, raw):
    design = _design(reference)
    weight = np.ones(len(reference))
    # A spike corrupts the entire rosette row. Share its weight across channels.
    for _ in range(24):
        coeff = np.linalg.lstsq(design * weight[:, None], raw * weight[:, None],
                                rcond=None)[0]
        error = np.linalg.norm(design @ coeff - raw, axis=1)
        scale = max(np.median(error), 1e-12)
        weight = np.sqrt(np.minimum(1., 1.5 * scale / np.maximum(error, 1e-12)))
    clean = error <= 8 * scale
    d, y = design[clean], raw[clean]
    full = np.linalg.lstsq(d, y, rcond=None)[0]

    # Prefer the observed cyclic interaction law when the rejected terms do
    # not improve the retained rows beyond their measurement scatter.
    compact = np.zeros_like(full)
    for channel, columns in enumerate(_CYCLIC):
        compact[columns, channel] = np.linalg.lstsq(d[:, columns], y[:, channel],
                                                   rcond=None)[0]
    if np.linalg.norm(d @ compact - y) <= 1.5 * np.linalg.norm(d @ full - y) + 1e-11:
        return compact
    return full


def _invert(raw, coeff):
    raw = np.asarray(raw, dtype=float)
    target = raw.reshape(-1, 4)
    x = np.linalg.solve(coeff[1:5].T, (target - coeff[0]).T).T
    derivative = (_DERIVATIVE @ coeff).transpose(1, 2, 0)
    tolerance = 1e-11 * (1 + np.max(np.abs(target), axis=1))
    for _ in range(32):
        design = _design(x)
        residual = design @ coeff - target
        active = np.flatnonzero(np.max(np.abs(residual), axis=1) > tolerance)
        if not len(active):
            break
        base, err = x[active], residual[active]
        jac = np.einsum('nt,tij->nij', design[active], derivative)
        step = np.linalg.solve(jac, err[..., None])[..., 0]
        base_error = np.linalg.norm(err, axis=1)
        trial_scale = np.ones(len(active))
        for _ in range(20):
            trial = base - trial_scale[:, None] * step
            improved = np.linalg.norm(_design(trial) @ coeff - target[active], axis=1) < base_error
            x[active[improved]] = trial[improved]
            if improved.all():
                break
            keep = ~improved
            active, base, step, base_error = active[keep], base[keep], step[keep], base_error[keep]
            trial_scale = 0.5 * trial_scale[keep]
    return x.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a study in reference coordinates without changing the input."""
    sweeps = study['sensor_calibration_sweeps']
    reference = np.asarray(sweeps['ref_rosette'], dtype=float)
    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
    coeff = _fit(reference, raw)
    banks = sweeps['single_bank_sweeps']
    calibrated = dict(study, y_obs_raw=_invert(study['y_obs_raw'], coeff),
                      sensor_calibration_sweeps=dict(
                          sweeps, raw_rosette=sweeps['ref_rosette'].copy(),
                          single_bank_sweeps=dict(banks, raw_readout=banks['ref_signal'].copy())))
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        calibrated['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coeff))
            if pulse['y_pulse_raw'] is not None else dict(pulse) for pulse in pulses]
    return calibrated
