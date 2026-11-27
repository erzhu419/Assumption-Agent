"""Fit and invert the coupled four-channel rosette response."""
import numpy as np


_BANKS = np.arange(4)
_NEXT = (_BANKS + 1) % 4
_PREV = (_BANKS - 1) % 4


def _features(x, j):
    u = x[:, j]
    v = x[:, _NEXT[j]]
    return np.column_stack((np.ones(len(x)), u, v, x[:, _PREV[j]], u * v, u**3))


def _fit(ref, raw, banks):
    # Gross EMI affects whole rosette rows. Screen against the typical row
    # amplitude before regression, then trim residual outliers jointly.
    amplitude = np.max(np.abs(raw), axis=1)
    keep = amplitude < 6 * np.median(amplitude)
    bank_id = banks['bank_id']
    bank_ref = np.eye(4)[bank_id] * banks['ref_signal'][:, None]
    bank_raw = banks['raw_readout']
    bank_keep = np.abs(bank_raw) < 6 * np.median(amplitude)
    features = [_features(ref, j) for j in _BANKS]
    bank_features = [_features(bank_ref, j) for j in _BANKS]
    for _ in range(8):
        coef = []
        for j in _BANKS:
            use = bank_keep & (bank_id == j)
            a = np.vstack((features[j][keep], bank_features[j][use]))
            b = np.r_[raw[keep, j], bank_raw[use]]
            coef.append(np.linalg.lstsq(a, b, rcond=None)[0])
        coef = np.array(coef)
        row_error = np.max(np.abs(_forward(ref, coef) - raw), axis=1)
        bank_error = np.abs(_forward(bank_ref, coef)[np.arange(len(bank_id)), bank_id] - bank_raw)
        threshold = max(8 * np.median(row_error[keep]), 1e-8)
        next_keep = keep & (row_error <= threshold)
        next_bank = bank_keep & (bank_error <= threshold)
        if np.array_equal(next_keep, keep) and np.array_equal(next_bank, bank_keep):
            break
        keep, bank_keep = next_keep, next_bank
    return coef


def _forward(x, coef):
    return np.column_stack([_features(x, j) @ coef[j] for j in _BANKS])


def _inverse(values, coef):
    raw = np.asarray(values, dtype=float).reshape(-1, 4)
    linear = np.zeros((4, 4))
    linear[_BANKS, _BANKS] = coef[:, 1]
    linear[_BANKS, _NEXT] = coef[:, 2]
    linear[_BANKS, _PREV] = coef[:, 3]
    x = np.linalg.solve(linear, (raw - coef[:, 0]).T).T
    for _ in range(40):
        residual = _forward(x, coef) - raw
        error = np.max(np.abs(residual), axis=1)
        active = error > 2e-12 * (1 + np.max(np.abs(raw), axis=1))
        if not np.any(active):
            break
        z = x[active]
        jacobian = np.broadcast_to(linear, (len(z), 4, 4)).copy()
        jacobian[:, _BANKS, _BANKS] += coef[:, 4] * z[:, _NEXT] + 3 * coef[:, 5] * z**2
        jacobian[:, _BANKS, _NEXT] += coef[:, 4] * z
        step = np.linalg.solve(jacobian, residual[active, :, None])[..., 0]
        target = raw[active]
        old_error = error[active]
        for _ in range(16):
            trial = z - step
            worse = np.max(np.abs(_forward(trial, coef) - target), axis=1) >= old_error
            if not np.any(worse):
                break
            step[worse] *= 0.5
        trial[worse] = z[worse]
        x[active] = trial
    return x.reshape(np.shape(values))


def calibrate_upstream(study: dict) -> dict:
    """Return a copy of the study in canonical observation coordinates."""
    sweep = study['sensor_calibration_sweeps']
    bank = sweep['single_bank_sweeps']
    coef = _fit(sweep['ref_rosette'], sweep['raw_rosette'], bank)
    out = dict(study, y_obs_raw=_inverse(study['y_obs_raw'], coef))
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        out['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_inverse(pulse['y_pulse_raw'], coef)) for pulse in pulses
        ]
    out['sensor_calibration_sweeps'] = dict(
        sweep, raw_rosette=sweep['ref_rosette'].copy(),
        single_bank_sweeps=dict(bank, raw_readout=bank['ref_signal'].copy()),
    )
    return out
