"""Fit and invert the four-channel rosette transfer function."""
import numpy as np
from numpy.polynomial.chebyshev import chebvander, chebval, chebder
from scipy.optimize import least_squares

_OPPOSITE = (2, 3, 0, 1)


def _basis(x, scale, k):
    j = _OPPOSITE[k]
    diagonal = chebvander(x[:, k] / scale[k], 9)
    coupled = chebvander(x[:, j] / scale[j], 7)[:, 1:]
    return np.column_stack((diagonal, coupled - chebvander(0., 7)[:, 1:]))


def _gross_mask(y):
    # EMI is a row event; do not let one affected channel enter another fit.
    if y.ndim == 1:
        distance = np.abs(y - np.median(y))
    else:
        distance = np.max(np.abs(y - np.median(y, axis=0)), axis=1)
    return distance <= 7 * max(np.median(distance), 1e-10)


def _fit(sweeps):
    x = np.asarray(sweeps['ref_rosette'], dtype=float)
    y = np.asarray(sweeps['raw_rosette'], dtype=float)
    scale = np.maximum(np.max(np.abs(x), axis=0), 1e-8)
    bank = sweeps['single_bank_sweeps']
    ids = np.asarray(bank['bank_id'], dtype=int)
    ref = np.asarray(bank['ref_signal'], dtype=float)
    raw = np.asarray(bank['raw_readout'], dtype=float)
    scale = np.maximum(scale, np.max(np.abs(ref)))
    axial = np.eye(4)[ids] * ref[:, None]
    keep = _gross_mask(y)
    bank_ok = _gross_mask(raw)
    designs = [_basis(x, scale, k) for k in range(4)]
    anchors = [_basis(axial, scale, k) for k in range(4)]
    bank_rows = [bank_ok & (ids == k) for k in range(4)]
    for _ in range(5):
        coeff = []
        for k in range(4):
            m = bank_rows[k]
            z = np.vstack((designs[k][keep], anchors[k][m]))
            v = np.r_[y[keep, k], raw[m]]
            coeff.append(np.linalg.lstsq(z, v, rcond=None)[0])
        pred = np.column_stack([z @ b for z, b in zip(designs, coeff)])
        error = np.max(np.abs(pred - y), axis=1)
        middle = np.median(error[keep])
        limit = max(0.008, 10 * middle)
        new = keep & (error <= limit)
        if np.array_equal(new, keep) or new.sum() < 24:
            break
        keep = new
    bounds = np.array([np.minimum(x.min(axis=0), [ref[ids == k].min() for k in range(4)]),
                       np.maximum(x.max(axis=0), [ref[ids == k].max() for k in range(4)])])
    return scale, coeff, bounds


def _forward(x, model):
    scale, coeff, bounds = model
    x = np.atleast_2d(x)
    t = np.clip(x, *bounds)
    y = np.empty_like(x)
    jac = np.zeros((len(x), 4, 4))
    for k, b in enumerate(coeff):
        j = _OPPOSITE[k]
        a, c = b[:10], np.r_[0., b[10:]]
        u, v = t[:, k] / scale[k], t[:, j] / scale[j]
        da, dc = chebval(u, chebder(a)) / scale[k], chebval(v, chebder(c)) / scale[j]
        da = np.where(t[:, k] != x[:, k], np.maximum(da, .1), da)
        y[:, k] = (chebval(u, a) + chebval(v, c) - chebval(0., c)
                   + da * (x[:, k] - t[:, k]) + dc * (x[:, j] - t[:, j]))
        jac[:, k, k], jac[:, k, j] = da, dc
    return y, jac


def _inverse(raw, model):
    data = np.asarray(raw, dtype=float)
    target = data.reshape(-1, 4)
    if not len(target):
        return data.copy()
    origin, jac0 = _forward(np.zeros((1, 4)), model)
    x = np.linalg.lstsq(jac0[0], (target - origin).T, rcond=None)[0].T
    for _ in range(30):
        val, jac = _forward(x, model)
        residual = val - target
        err = np.max(np.abs(residual), axis=1)
        if np.max(err) < 2e-11:
            break
        try:
            step = np.linalg.solve(jac, residual[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            break
        step /= np.maximum(1., np.max(np.abs(step), axis=1, keepdims=True))
        for rate in (1., .5, .25, .125):
            trial = x - rate * step
            alt = np.max(np.abs(_forward(trial, model)[0] - target), axis=1)
            accept = alt < err
            x[accept] = trial[accept]
            err[accept] = alt[accept]
            step[accept] = 0.
        if not np.any(accept) and np.max(np.abs(step)) == 0.:
            break
    bad = np.max(np.abs(_forward(x, model)[0] - target), axis=1) > 1e-8
    for i in np.flatnonzero(bad):
        x[i] = least_squares(lambda q: _forward(q, model)[0][0] - target[i], x[i],
                             jac=lambda q: _forward(q, model)[1][0], max_nfev=80).x
    return x.reshape(data.shape)


def calibrate_upstream(study: dict) -> dict:
    sweeps = study['sensor_calibration_sweeps']
    model = _fit(sweeps)
    out = dict(study, y_obs_raw=_inverse(study['y_obs_raw'], model))
    out['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=np.asarray(sweeps['ref_rosette']).copy(),
        single_bank_sweeps=dict(sweeps['single_bank_sweeps'],
                                raw_readout=np.asarray(sweeps['single_bank_sweeps']['ref_signal']).copy()))
    if study.get('reference_tracer_pulses') is not None:
        out['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_inverse(pulse['y_pulse_raw'], model))
            if pulse is not None and pulse.get('y_pulse_raw') is not None else pulse
            for pulse in study['reference_tracer_pulses']]
    return out
