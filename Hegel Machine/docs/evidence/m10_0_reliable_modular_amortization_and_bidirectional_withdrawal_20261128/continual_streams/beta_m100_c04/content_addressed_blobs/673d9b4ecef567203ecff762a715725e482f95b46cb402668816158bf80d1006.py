"""Robust calibration of the four-channel rosette readout."""

import numpy as np
from scipy.optimize import least_squares

_PAIR = np.array([2, 3, 0, 1])


def _basis(x, freq):
    paired = x[..., _PAIR]
    return np.stack((np.ones_like(x), x, paired, np.sin(freq[0] * x),
                     1. - np.cos(freq[1] * paired)), axis=-1)


def _fit(ref, raw):
    # Reject gross EMI before fitting; downweight remaining rows jointly across channels.
    center = np.median(raw, axis=0)
    scale = np.maximum(np.median(abs(raw - center), axis=0), 0.25)
    good = np.max(abs(raw - center) / scale, axis=1) < 12.
    x, y = ref[good], raw[good]
    weight = np.ones(len(x))
    freq = np.array([2.5, 2.])

    def linear_fit(q):
        design = _basis(x, q)
        coef = np.array([np.linalg.lstsq(design[:, j] * weight[:, None],
                                        y[:, j] * weight, rcond=None)[0]
                         for j in range(4)])
        return coef, np.einsum('njk,jk->nj', design, coef) - y

    for _ in range(8):
        freq = least_squares(lambda q: (linear_fit(q)[1] * weight[:, None]).ravel(),
                             freq, bounds=([1.5, 1.], [3.5, 3.]),
                             ftol=1e-10, gtol=1e-12).x
        coef, residual = linear_fit(freq)
        row_error = np.sqrt(np.mean(residual**2, axis=1))
        cutoff = max(4. * np.median(row_error), 3e-5)
        updated = 1. / np.hypot(1., row_error / cutoff)
        if np.max(abs(updated - weight)) < 1e-4:
            break
        weight = updated
    return coef, freq


def _response(x, model):
    coef, freq = model
    value = np.einsum('...jk,jk->...j', _basis(x, freq), coef)
    diag = coef[:, 1] + freq[0] * coef[:, 3] * np.cos(freq[0] * x)
    cross = coef[:, 2] + freq[1] * coef[:, 4] * np.sin(freq[1] * x[..., _PAIR])
    jac = np.zeros(x.shape[:-1] + (4, 4))
    jac[..., np.arange(4), np.arange(4)] = diag
    jac[..., np.arange(4), _PAIR] = cross
    return value, jac


def _invert(raw, model):
    y = np.asarray(raw, dtype=float)
    targets = y.reshape(-1, 4)
    base, slope = _response(np.zeros(4), model)
    x = np.linalg.solve(slope, (targets - base).T).T
    for _ in range(24):
        value, jac = _response(x, model)
        err = value - targets
        size = np.max(abs(err), axis=1)
        if np.max(size, initial=0.) <= 2e-12:
            break
        # A row with poor conditioning must not stop inversion of other rows.
        delta = (np.linalg.pinv(jac) @ err[..., None])[..., 0]
        delta /= np.maximum(1., np.max(abs(delta), axis=1, keepdims=True))
        accepted = size <= 2e-12
        for step in (1., .5, .25, .125, .0625):
            trial = x - step * delta
            improve = (~accepted) & (np.max(abs(_response(trial, model)[0] - targets), axis=1) < size)
            x[improve] = trial[improve]
            accepted |= improve
            if np.all(accepted):
                break
        if not np.any(accepted & (size > 2e-12)):
            break
    for i in np.flatnonzero(np.max(abs(_response(x, model)[0] - targets), axis=1) > 1e-9):
        r = least_squares(lambda z: _response(z, model)[0] - targets[i], x[i],
                          jac=lambda z: _response(z, model)[1])
        x[i] = r.x
    return x.reshape(y.shape)


def calibrate_upstream(study: dict) -> dict:
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
    model = None if np.array_equal(raw, ref) else _fit(ref, raw)

    def convert(value):
        return np.array(value, copy=True) if model is None else _invert(value, model)

    result = dict(study, y_obs_raw=convert(study['y_obs_raw']))
    bank = sweeps['single_bank_sweeps']
    result['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=np.array(sweeps['ref_rosette'], copy=True),
        single_bank_sweeps=dict(bank, raw_readout=np.array(bank['ref_signal'], copy=True)))
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        result['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=convert(pulse['y_pulse_raw']))
            if pulse is not None and pulse.get('y_pulse_raw') is not None else pulse
            for pulse in pulses]
    return result
