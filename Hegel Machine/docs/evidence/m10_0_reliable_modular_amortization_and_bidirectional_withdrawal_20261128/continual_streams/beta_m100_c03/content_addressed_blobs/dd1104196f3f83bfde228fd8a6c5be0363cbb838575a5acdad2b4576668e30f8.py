"""Fit and invert the four-channel rosette transfer function."""
import numpy as np
from numpy.polynomial.chebyshev import chebvander, chebval, chebder
from scipy.optimize import least_squares

_OPPOSITE = (2, 3, 0, 1)


def _diagonal(t, p):
    a, b, c, d, e = p
    q = t*t
    numerator = t * (1 + q*(a + b*q))
    denominator = 1 + q*(c + q*(d + e*q))
    slope = ((1 + q*(3*a + 5*b*q))*denominator
             - numerator*2*t*(c + q*(2*d + 3*e*q))) / denominator**2
    return numerator / denominator, slope


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
    bank = sweeps['single_bank_sweeps']
    ids = np.asarray(bank['bank_id'], dtype=int)
    ref = np.asarray(bank['ref_signal'], dtype=float)
    raw = np.asarray(bank['raw_readout'], dtype=float)
    keep, bank_ok = _gross_mask(y), _gross_mask(raw)
    # The independent axial sweeps constrain the diagonal response; joint
    # rows constrain the opposite-bank response in the same normalized units.
    bank_id = np.r_[np.tile(np.arange(4), len(x)), ids]
    own = np.r_[x.ravel(), ref]
    other = np.r_[x[:, _OPPOSITE].ravel(), np.zeros(len(ref))]
    readout = np.r_[y.ravel(), raw]
    scale = max(np.max(np.abs(x)), np.max(np.abs(ref)))
    coupling = chebvander(other / scale, 9)[:, 1:] - chebvander(0., 9)[:, 1:]
    groups = np.eye(4)[bank_id]
    params = np.r_[[-.19, .066, .21, .024, .0024], np.zeros(9)]
    low, high = np.r_[[-1, 0, 0, 0, 0], [-1]*9], np.r_[[1, 1, 2, 2, 1], [1]*9]
    for _ in range(4):
        selected = np.r_[np.repeat(keep, 4), bank_ok]
        def profile(p, full=False):
            shape = _diagonal(own, p[:5])[0] + coupling @ p[5:]
            linear = np.column_stack((groups, groups * shape[:, None]))
            coef = np.linalg.lstsq(linear[selected], readout[selected], rcond=None)[0]
            residual = linear @ coef - readout
            return (coef, residual) if full else residual[selected]

        params = least_squares(profile, params, bounds=(low, high),
                               max_nfev=100, gtol=1e-11).x
        coef, residual = profile(params, True)
        row_error = np.max(np.abs(residual[:y.size].reshape(y.shape)), axis=1)
        bank_error = np.abs(residual[y.size:])
        limit = max(.003, 10*np.median(np.r_[row_error[keep], bank_error[bank_ok]]))
        new_keep = keep & (row_error <= limit)
        new_bank = bank_ok & (bank_error <= limit)
        if (np.array_equal(new_keep, keep) and np.array_equal(new_bank, bank_ok)
                or new_keep.sum() < 24):
            break
        keep, bank_ok = new_keep, new_bank
    # Leave room for interpolation at a held-out sweep endpoint, but avoid
    # unconstrained polynomial extrapolation far outside the sweep support.
    bounds = np.array([min(x.min(), ref.min())-.08, max(x.max(), ref.max())+.08])
    return scale, params, coef, bounds


def _forward(x, model):
    scale, params, coef, bounds = model
    x = np.atleast_2d(x)
    clipped = np.clip(x, *bounds)
    diagonal, dd = _diagonal(clipped, params[:5])
    c = np.r_[0., params[5:]]
    u = clipped / scale
    cross = chebval(u, c) - chebval(0., c)
    dc = chebval(u, chebder(c)) / scale
    diag = diagonal + dd * (x - clipped)
    cross = cross + dc * (x - clipped)
    gain = coef[4:]
    y = coef[:4] + gain * (diag + cross[:, _OPPOSITE])
    jac = np.zeros((len(x), 4, 4))
    jac[:, np.arange(4), np.arange(4)] = dd * gain
    jac[:, np.arange(4), _OPPOSITE] = dc[:, _OPPOSITE] * gain
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
        moved = False
        for rate in (1., .5, .25, .125):
            trial = x - rate * step
            alt = np.max(np.abs(_forward(trial, model)[0] - target), axis=1)
            accept = alt < err
            moved |= bool(np.any(accept))
            x[accept] = trial[accept]
            err[accept] = alt[accept]
            step[accept] = 0.
        if not moved:
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
