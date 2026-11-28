"""Calibrate the coupled rosette before the canonical transport solve."""
from __future__ import annotations

import numpy as np


_CHANNELS = np.arange(4)
_NEXT = (_CHANNELS + 1) % 4


def _design(ref, channel):
    own = ref[:, channel]
    return np.column_stack((np.ones(len(ref)), ref,
                            own * ref[:, _NEXT[channel]], own**3))


def _fit(ref, raw, bank):
    """Fit cyclic quadratic coupling plus diagonal cubic curvature.

    Rosette EMI is a row event, so the four residuals share a robust weight.
    A single-bank readout contributes only to the channel actually measured.
    """
    bid = np.asarray(bank['bank_id'])
    bank_ref = np.asarray(bank['ref_signal'])
    bank_raw = np.asarray(bank['raw_readout'])
    design = [_design(ref, j) for j in _CHANNELS]
    single_design, single_raw = [], []
    for j in _CHANNELS:
        rows = bid == j
        single_ref = np.zeros((np.count_nonzero(rows), 4))
        single_ref[:, j] = bank_ref[rows]
        single_design.append(_design(single_ref, j))
        single_raw.append(bank_raw[rows])

    wr = np.ones(len(ref))
    wb = [np.ones(len(y)) for y in single_raw]
    coef = np.empty((7, 4))
    for _ in range(18):
        for j, (a, b, y, w) in enumerate(zip(design, single_design, single_raw, wb)):
            lhs = np.vstack((a * wr[:, None], b * w[:, None]))
            rhs = np.concatenate((raw[:, j] * wr, y * w))
            coef[:, j] = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        row_error = np.linalg.norm(_forward(ref, coef) - raw, axis=1)
        scale = max(np.median(row_error) / 1.83, 1e-8)
        wr = 1 / np.hypot(1, row_error / (4 * scale))
        wb = [1 / np.hypot(1, (b @ coef[:, j] - y) / (3 * scale))
              for j, (b, y) in enumerate(zip(single_design, single_raw))]
    return coef


def _forward(ref, coef):
    return (coef[0] + ref @ coef[1:5]
            + coef[5] * ref * ref[:, _NEXT] + coef[6] * ref**3)


def _inverse(raw, coef):
    original = np.asarray(raw)
    target = np.asarray(original, dtype=float).reshape(-1, 4)
    ref = np.linalg.solve(coef[1:5].T, (target - coef[0]).T).T
    tol = 1e-12 * (1 + np.max(np.abs(target), initial=0.0))
    for _ in range(40):
        residual = _forward(ref, coef) - target
        error = np.max(np.abs(residual), axis=1)
        active = error > tol
        if not active.any():
            return ref.reshape(original.shape)
        x = ref[active]
        jac = np.broadcast_to(coef[1:5].T, (len(x), 4, 4)).copy()
        jac[:, _CHANNELS, _CHANNELS] += coef[5] * x[:, _NEXT] + 3 * coef[6] * x**2
        jac[:, _CHANNELS, _NEXT] += coef[5] * x
        step = np.linalg.solve(jac, residual[active, :, None])[..., 0]
        old_norm = np.linalg.norm(residual[active], axis=1)
        trial = x - step
        for _ in range(20):
            worse = np.linalg.norm(_forward(trial, coef) - target[active], axis=1) >= old_norm
            if not worse.any():
                break
            step[worse] *= 0.5
            trial[worse] = x[worse] - step[worse]
        ref[active] = trial
    raise RuntimeError('Rosette inverse did not converge')


def calibrate_upstream(study: dict) -> dict:
    """Return a canonical-coordinate study without modifying the input."""
    sweeps = study['sensor_calibration_sweeps']
    bank = sweeps['single_bank_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
    coef = _fit(ref, raw, bank)
    out = dict(study)
    out['y_obs_raw'] = _inverse(study['y_obs_raw'], coef)
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        out['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_inverse(pulse['y_pulse_raw'], coef))
            if pulse['y_pulse_raw'] is not None else dict(pulse)
            for pulse in pulses
        ]
    out['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=np.asarray(sweeps['ref_rosette']).copy(),
        single_bank_sweeps=dict(bank, raw_readout=np.asarray(bank['ref_signal']).copy()),
    )
    return out
