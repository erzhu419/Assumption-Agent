"""Robust forward calibration and inversion of the four-channel rosette."""

import numpy as np


def _fit_forward(ref, raw, bank):
    """Fit offset, gain, cyclic neighbors, pair interaction, and self cubic."""
    rosette_design = np.stack((np.ones_like(ref), ref, np.roll(ref, 1, axis=1),
                               np.roll(ref, -1, axis=1),
                               ref * np.roll(ref, -1, axis=1), ref**3), axis=-1)
    bank_id = np.asarray(bank['bank_id'])
    signal = np.asarray(bank['ref_signal'], dtype=float)
    readout = np.asarray(bank['raw_readout'], dtype=float)
    designs, targets = [], []
    for j in range(4):
        take = bank_id == j
        t = signal[take]
        single = np.column_stack((np.ones_like(t), t, np.zeros((len(t), 3)), t**3))
        designs.append(np.vstack((rosette_design[:, j, :], single)))
        targets.append(np.r_[raw[:, j], readout[take]])

    n = len(ref)
    weights = [np.ones(len(y)) for y in targets]
    for _ in range(24):
        coeff = np.stack([np.linalg.lstsq(d * w[:, None], y * w, rcond=None)[0]
                          for d, y, w in zip(designs, targets, weights)])
        errors = [y - d @ c for d, y, c in zip(designs, targets, coeff)]
        row_error = np.linalg.norm(np.column_stack([e[:n] for e in errors]), axis=1)
        # An EMI excursion invalidates the joint row; single-bank samples are separate.
        weights = [1 / np.hypot(1, np.r_[row_error, e[n:]] / 0.02) for e in errors]

    # Once gross excursions are separated, remove their residual influence.
    good_rows = row_error < max(0.005, 8 * np.median(row_error))
    keep = [np.r_[good_rows, np.abs(e[n:]) < max(0.005, 8 * np.median(row_error))]
            for e in errors]
    return np.stack([np.linalg.lstsq(d[k], y[k], rcond=None)[0]
                     for d, y, k in zip(designs, targets, keep)])


def _forward(x, coeff):
    b, gain, prev, nxt, pair, cubic = coeff.T
    next_x = np.roll(x, -1, axis=1)
    return b + gain*x + prev*np.roll(x, 1, axis=1) + nxt*next_x + pair*x*next_x + cubic*x**3


def _inverse(raw, coeff):
    """Use the coupled linear response to initialize damped Newton steps."""
    values = np.asarray(raw, dtype=float)
    y = values.reshape(-1, 4)
    b, gain, prev, nxt, pair, cubic = coeff.T
    j = np.arange(4)
    lin = np.diag(gain)
    lin[j, (j-1) % 4] = prev
    lin[j, (j+1) % 4] = nxt
    x = np.linalg.solve(lin, (y-b).T).T
    for _ in range(32):
        residual = _forward(x, coeff) - y
        active = np.max(np.abs(residual), axis=1) > 1e-12
        if not active.any():
            return x.reshape(values.shape)
        u, r = x[active], residual[active]
        jac = np.broadcast_to(lin, (len(u), 4, 4)).copy()
        jac[:, j, j] += pair*np.roll(u, -1, axis=1) + 3*cubic*u*u
        jac[:, j, (j+1) % 4] += pair*u
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        old = np.sum(r*r, axis=1)
        trial = u-step
        for _ in range(16):
            err = _forward(trial, coeff) - y[active]
            backtrack = np.sum(err*err, axis=1) >= old
            if not backtrack.any():
                break
            step[backtrack] *= 0.5
            trial[backtrack] = u[backtrack]-step[backtrack]
        x[active] = trial
    raise RuntimeError('Coupled rosette inversion did not converge')


def calibrate_upstream(study: dict) -> dict:
    sweeps = study['sensor_calibration_sweeps']
    bank = sweeps['single_bank_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
    coeff = _fit_forward(ref, raw, bank)
    calibrated = dict(study)
    calibrated['y_obs_raw'] = _inverse(study['y_obs_raw'], coeff)
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        calibrated['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_inverse(pulse['y_pulse_raw'], coeff))
            for pulse in pulses
        ]
    calibrated['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=sweeps['ref_rosette'].copy(),
        single_bank_sweeps=dict(bank, raw_readout=bank['ref_signal'].copy()))
    return calibrated
