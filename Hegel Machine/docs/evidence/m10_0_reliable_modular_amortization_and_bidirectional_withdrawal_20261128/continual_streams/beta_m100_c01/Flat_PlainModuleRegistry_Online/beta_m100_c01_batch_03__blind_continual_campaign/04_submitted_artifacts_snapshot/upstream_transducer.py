"""Fit and invert the coupled four-channel rosette calibration."""

import numpy as np
from scipy.optimize import least_squares


def _design(x, j):
    own = x[:, j]
    nxt = x[:, (j + 1) % 4]
    prev = x[:, (j - 1) % 4]
    return np.column_stack((np.ones(len(x)), own, nxt, prev, own * nxt, own**3))


def _fit(sweeps):
    x = np.asarray(sweeps['ref_rosette'], dtype=float)
    y = np.asarray(sweeps['raw_rosette'], dtype=float)
    affine = np.column_stack((np.ones(len(x)), x))
    weight = np.ones(len(x))
    # Downweight whole rows: the observed EMI shifts several channels together.
    for _ in range(12):
        p = np.linalg.lstsq(affine * weight[:, None], y * weight[:, None], rcond=None)[0]
        residual = np.linalg.norm(affine @ p - y, axis=1)
        scale = max(np.median(residual), 1e-8)
        weight = 1 / np.hypot(1, residual / (2 * scale))
    keep = residual <= 8 * scale
    banks = sweeps['single_bank_sweeps']
    ids = np.asarray(banks['bank_id'])
    ref = np.asarray(banks['ref_signal'], dtype=float)
    raw = np.asarray(banks['raw_readout'], dtype=float)
    axes = ref[:, None] * (ids[:, None] == np.arange(4))
    design = [_design(x, j) for j in range(4)]
    p = np.stack([np.linalg.lstsq(a[keep], y[keep, j], rcond=None)[0]
                  for j, a in enumerate(design)])
    # Re-evaluate rows after accounting for the known nonlinear response.
    residual = np.linalg.norm(np.column_stack([a @ q for a, q in zip(design, p)]) - y, axis=1)
    scale = max(np.median(residual), 1e-8)
    keep = residual <= 8 * scale
    for j, a in enumerate(design):
        bank = ids == j
        b = _design(axes[bank], j)
        readout = raw[bank]
        good = np.abs(b @ p[j] - readout) <= 8 * scale
        p[j] = np.linalg.lstsq(np.vstack((a[keep], b[good])),
                               np.concatenate((y[keep, j], readout[good])), rcond=None)[0]
    return p


def _forward(x, p):
    off, main, forward, backward, pair, cubic = p.T
    nxt = np.roll(x, -1, axis=-1)
    prev = np.roll(x, 1, axis=-1)
    return off + main * x + forward * nxt + backward * prev + pair * x * nxt + cubic * x**3


def _jacobian(x, p):
    _, main, forward, backward, pair, cubic = p.T
    j = np.arange(4)
    result = np.zeros(x.shape[:-1] + (4, 4))
    result[..., j, j] = main + pair * np.roll(x, -1, axis=-1) + 3 * cubic * x**2
    result[..., j, (j + 1) % 4] = forward + pair * x
    result[..., j, (j - 1) % 4] = backward
    return result


def _invert(raw, p):
    array = np.asarray(raw, dtype=float)
    y = array.reshape(-1, 4)
    x = np.linalg.solve(_jacobian(np.zeros(4), p), (y - p[:, 0]).T).T
    # Batch Newton with per-row backtracking for large tracer-pulse amplitudes.
    for _ in range(20):
        r = _forward(x, p) - y
        error = np.linalg.norm(r, axis=1)
        active = error > 1e-11
        if not np.any(active):
            break
        a = x[active]
        jac = _jacobian(a, p)
        step = np.linalg.solve(jac, r[active, :, None])[..., 0]
        target = y[active]
        trial = a - step
        old_error = error[active]
        for _ in range(12):
            bad = np.linalg.norm(_forward(trial, p) - target, axis=1) >= old_error
            if not np.any(bad):
                break
            step[bad] *= 0.5
            trial[bad] = a[bad] - step[bad]
        good = np.linalg.norm(_forward(trial, p) - target, axis=1) < old_error
        x[np.flatnonzero(active)[good]] = trial[good]
    for j in np.flatnonzero(np.linalg.norm(_forward(x, p) - y, axis=1) > 1e-9):
        x[j] = least_squares(lambda z: _forward(z, p) - y[j], x[j],
                             jac=lambda z: _jacobian(z, p)).x
    return x.reshape(array.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a canonical-coordinate study without changing the input study."""
    sweeps = study['sensor_calibration_sweeps']
    banks = sweeps['single_bank_sweeps']
    p = _fit(sweeps)
    calibrated = dict(study)
    calibrated['y_obs_raw'] = _invert(study['y_obs_raw'], p)
    calibrated['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=sweeps['ref_rosette'].copy(),
        single_bank_sweeps=dict(banks, raw_readout=banks['ref_signal'].copy()),
    )
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        calibrated['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], p))
            for pulse in pulses
        ]
    return calibrated
