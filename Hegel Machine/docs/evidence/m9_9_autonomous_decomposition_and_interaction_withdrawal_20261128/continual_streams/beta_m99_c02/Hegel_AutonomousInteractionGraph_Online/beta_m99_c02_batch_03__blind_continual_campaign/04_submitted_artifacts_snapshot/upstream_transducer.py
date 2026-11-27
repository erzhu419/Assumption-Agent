"""Calibrate a four-bank rosette before the canonical transport inversion."""

import numpy as np

_BANK = np.arange(4)
_NEXT = (_BANK + 1) % 4
_PREV = (_BANK - 1) % 4


def _terms(x):
    neighbor = x[:, _NEXT]
    return np.stack((neighbor, x[:, _PREV], x**3, x * neighbor), axis=-1)


def _fit(x, y):
    """Fit offsets and gains per bank, sharing the normalized cyclic law."""
    terms = _terms(x)
    design = np.concatenate((np.ones_like(x)[..., None], x[..., None], terms), axis=2)
    # Scale by the size of the excitation before using raw magnitudes to seed
    # the fit: an ordinary high-amplitude cubic response is not an EMI spike.
    strength = np.max(np.abs(y), axis=1) / (1 + np.max(np.abs(x), axis=1)**3)
    w = (1 + (strength / (4 * np.median(strength)))**4)**-1
    root = np.sqrt(w)
    coef = np.array([np.linalg.lstsq(design[:, j] * root[:, None], y[:, j] * root,
                                    rcond=None)[0] for j in _BANK])
    q = np.median(coef[:, 2:] / coef[:, 1, None], axis=0)
    for _ in range(400):
        previous = q.copy()
        h = x + terms @ q
        h0 = h - np.average(h, weights=w, axis=0)
        gain = np.sum(w[:, None] * h0 * y, axis=0) / np.sum(w[:, None] * h0**2, axis=0)
        offset = np.average(y - h * gain, weights=w, axis=0)
        root = np.sqrt(w)[:, None]
        q = np.linalg.lstsq((terms * (root * gain)[..., None]).reshape(-1, 4),
                            ((y - offset - gain * x) * root).ravel(), rcond=None)[0]
        residual = np.max(np.abs(y - offset - gain * (x + terms @ q)), axis=1)
        cutoff = 8 * max(np.median(residual), np.finfo(float).eps)
        updated = 1 / (1 + (residual / cutoff)**2)
        if np.max(np.abs(q - previous)) < 1e-11 and np.max(np.abs(updated - w)) < 1e-5:
            break
        w = updated
    return offset, gain, q


def _inverse(raw, offset, gain, q):
    shape = np.shape(raw)
    y = (np.asarray(raw, dtype=float).reshape(-1, 4) - offset) / gain
    x = np.zeros_like(y)
    if not len(x):
        return x.reshape(shape)
    jac = np.zeros((len(y), 4, 4))
    for _ in range(30):
        f = x + _terms(x) @ q - y
        err = np.max(np.abs(f), axis=1)
        if err.max() < 1e-12:
            break
        jac[:, _BANK, _BANK] = 1 + 3*q[2]*x*x + q[3]*x[:, _NEXT]
        jac[:, _BANK, _NEXT] = q[0] + q[3]*x
        jac[:, _BANK, _PREV] = q[1]
        step = np.linalg.solve(jac, f[..., None])[..., 0]
        trial = x - step
        for _ in range(20):
            worse = np.max(np.abs(trial + _terms(trial) @ q - y), axis=1) > err
            if not worse.any():
                break
            step[worse] *= 0.5
            trial[worse] = x[worse] - step[worse]
        x = trial
    return x.reshape(shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a calibrated study without modifying its original arrays."""
    sweeps = study['sensor_calibration_sweeps']
    ref = sweeps['ref_rosette']
    model = _fit(ref, sweeps['raw_rosette'])
    banks = sweeps['single_bank_sweeps']
    result = dict(study)
    result['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=ref.copy(),
        single_bank_sweeps=dict(banks, raw_readout=banks['ref_signal'].copy()))
    result['y_obs_raw'] = _inverse(study['y_obs_raw'], *model)
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        result['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_inverse(pulse['y_pulse_raw'], *model))
            for pulse in pulses
        ]
    return result
