"""Robust calibration of the coupled four-channel bridge."""
import numpy as np
from scipy.optimize import least_squares


_PARTNER = np.array([2, 3, 0, 1])


def _design(x, frequency):
    v = x[..., _PARTNER]
    s, t = frequency
    return np.stack((np.ones_like(x), x, np.sin(s*x), v, np.cos(t*v)-1), axis=-1)


def _fit(x, raw):
    """Profile the linear coefficients, with one outlier weight per bridge row."""
    center = np.median(raw, axis=0)
    scale = np.maximum(np.quantile(abs(raw-center), 0.75, axis=0), 1e-12)
    weight = (np.max(abs(raw-center)/scale, axis=1) < 6).astype(float)
    frequency = np.array([2.5, 2.0])

    def profile(f, w):
        a = _design(x, f)
        c = np.stack([np.linalg.lstsq(a[:, j]*w[:, None], raw[:, j]*w,
                                     rcond=None)[0] for j in range(4)])
        residual = (a*c).sum(axis=-1) - raw
        return c, residual

    for _ in range(6):
        w = np.sqrt(weight)
        frequency = least_squares(lambda f: (profile(f, w)[1]*w[:, None]).ravel(),
                                  frequency, bounds=([1.8, 1.2], [3.2, 2.8]),
                                  max_nfev=45).x
        coef, residual = profile(frequency, w)
        sigma = max(1.4826*np.median(abs(residual-np.median(residual, axis=0))), 1e-12)
        radius = np.sqrt(np.mean(residual**2, axis=1))
        weight = 1/(1+(radius/(3*sigma))**2)
    return frequency, coef


def _invert(raw, frequency, coef):
    """Iterate the paired coupling; bisect each monotone self-response."""
    raw = np.asarray(raw, dtype=float)
    offset, gain, sine, cross, cosine = coef.T
    s, t = frequency
    y = raw - offset
    x = y/gain
    width = abs(sine/gain) + 1e-12
    for _ in range(220):
        v = x[..., _PARTNER]
        target = y - cross*v - cosine*(np.cos(t*v)-1)
        center = target/gain
        lo, hi = center-width, center+width
        for _ in range(36):
            mid = (lo+hi)/2
            lower = gain*mid + sine*np.sin(s*mid) < target
            lo, hi = np.where(lower, mid, lo), np.where(lower, hi, mid)
        candidate = (lo+hi)/2
        if np.max(abs(candidate-x), initial=0.) < 3e-10:
            return candidate
        x = candidate
    return x


def calibrate_upstream(dataset: dict) -> dict:
    result = dict(dataset)
    cal = dataset['calibration_load_cells']
    true = np.asarray(cal['true_channel_loads'])
    raw = np.asarray(cal['bridge_raw'])
    model = None if np.array_equal(true, raw) else _fit(true, raw)
    if model is not None:
        for name in ('train', 'test', 'chamber_verification_trials'):
            part = dataset.get(name)
            if part is not None:
                result[name] = dict(part)
                for field in ('past_channel_raw', 'hold_channel_raw'):
                    result[name][field] = _invert(part[field], *model)
    result['calibration_load_cells'] = dict(cal, bridge_raw=true.copy())
    sweeps = cal['single_channel_sweeps']
    result['calibration_load_cells']['single_channel_sweeps'] = dict(
        sweeps, cmd_stress=np.asarray(sweeps['true_stress']).copy())
    return result
