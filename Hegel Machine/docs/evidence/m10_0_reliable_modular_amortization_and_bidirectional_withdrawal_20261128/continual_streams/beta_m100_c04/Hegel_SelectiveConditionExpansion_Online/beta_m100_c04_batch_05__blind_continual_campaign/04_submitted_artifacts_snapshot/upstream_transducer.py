"""Robust per-study calibration of the four-channel sensor rosette."""
import numpy as np
from scipy.optimize import least_squares


_PAIR = np.array([2, 3, 0, 1])


def _clean(values):
    center = np.median(values, axis=0)
    delta = np.abs(values - center)
    scale = np.maximum(np.median(delta, axis=0), 1e-10)
    return np.max(delta / scale, axis=1) < 9


def _basis(z, p, shape):
    a, b = shape
    return np.column_stack((np.ones_like(z), np.tanh(a*z), z, z**3, z**5,
                            p, np.tanh(b*p*p)))


def _fit(sweep):
    x = np.asarray(sweep['ref_rosette'], dtype=float)
    y = np.asarray(sweep['raw_rosette'], dtype=float)
    keep = _clean(y)
    x, y = x[keep], y[keep]
    bank = sweep['single_bank_sweeps']
    bank_id = np.asarray(bank['bank_id'])
    u = np.asarray(bank['ref_signal'], dtype=float)
    v = np.asarray(bank['raw_readout'], dtype=float)
    data = []
    for j, k in enumerate(_PAIR):
        uj, vj = u[bank_id == j], v[bank_id == j]
        keep = _clean(vj[:, None])
        uj, vj = uj[keep], vj[keep]
        data.append((np.r_[x[:, j], uj], np.r_[x[:, k], np.zeros_like(uj)],
                     np.r_[y[:, j], vj]))

    # Variable projection: solve the linear gains and offsets at each trial
    # saturation shape. The soft-L1 objective limits the influence of residual spikes on the shape fit.
    def profile(shape, coefficients=False):
        result = []
        for z, p, target in data:
            design = _basis(z, p, shape)
            coef = np.linalg.lstsq(design, target, rcond=None)[0]
            result.append(coef if coefficients else design @ coef - target)
        return np.array(result) if coefficients else np.concatenate(result)

    shape = least_squares(profile, [0.8, 0.9], bounds=([0.3, 0.05], [2.8, 3.0]),
                          loss='soft_l1', f_scale=0.008, max_nfev=100).x
    return shape, profile(shape, True)


def _forward(x, shape, coef):
    return np.column_stack([_basis(x[:, j], x[:, k], shape) @ coef[j]
                            for j, k in enumerate(_PAIR)])


def _inverse(raw, shape, coef):
    raw = np.asarray(raw, dtype=float)
    target = raw.reshape(-1, 4)
    a, b = shape
    offset, t, linear, cubic, quintic, cross, q = coef.T
    gain = np.diag(a*t + linear)
    gain[np.arange(4), _PAIR] = cross
    start = np.linalg.solve(gain, (target - offset).T).T

    def jac(z):
        diag = a*t*(1 - np.tanh(a*z)**2) + linear + 3*cubic*z*z + 5*quintic*z**4
        p = z[_PAIR]
        off = cross + 2*b*q*p*(1 - np.tanh(b*p*p)**2)
        mat = np.diag(diag)
        mat[np.arange(4), _PAIR] = off
        return mat

    for i, row in enumerate(target):
        start[i] = least_squares(lambda z: _forward(z[None, :], shape, coef)[0] - row,
                                 start[i], jac=jac, ftol=1e-11, xtol=1e-11,
                                 gtol=1e-11, max_nfev=80).x
    return start.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a study in canonical sensor coordinates without mutating input."""
    sweep = study['sensor_calibration_sweeps']
    shape, coef = _fit(sweep)
    bank = sweep['single_bank_sweeps']
    out = dict(study)
    out['sensor_calibration_sweeps'] = dict(
        sweep, raw_rosette=np.asarray(sweep['ref_rosette']).copy(),
        single_bank_sweeps=dict(bank, raw_readout=np.asarray(bank['ref_signal']).copy()),
    )
    out['y_obs_raw'] = _inverse(study['y_obs_raw'], shape, coef)
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        out['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_inverse(pulse['y_pulse_raw'], shape, coef))
            if pulse.get('y_pulse_raw') is not None else dict(pulse)
            for pulse in pulses
        ]
    return out
