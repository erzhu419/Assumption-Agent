"""Recover the shared coordinates from the two calibration panels."""
import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit_sensor(u, v):
    # The gross faults affect entire paired calibration rows. Screen them
    # against a robust diagonal fit before fitting the coupled response.
    i, j = np.triu_indices(len(u), 1)
    dx = u[j] - u[i]
    dy = v[j] - v[i]
    slopes = np.full_like(dx, np.nan)
    np.divide(dy, dx, out=slopes, where=np.abs(dx) > 0.1 * np.ptp(u, axis=0))
    slope = np.nanmedian(slopes, axis=0)
    bias = np.median(v - u * slope, axis=0)
    err = np.abs(v - (bias + u * slope))
    keep = np.all(err <= np.maximum(8 * np.median(err, axis=0), 1e-4), axis=1)
    if np.count_nonzero(keep) >= 8:
        u, v = u[keep], v[keep]
    paired = np.roll(u, 2, axis=1)
    d = np.zeros((len(u), 4, 10))
    k = np.arange(4)
    d[:, k, k] = 1
    d[:, k, 4 + k] = u

    def fit(freq, return_coeff=False):
        d[:, :, 8] = np.tanh(freq[0] * u)
        d[:, :, 9] = np.sin(freq[1] * paired)
        design = d.reshape(-1, 10)
        coeff = np.linalg.lstsq(design, v.ravel(), rcond=None)[0]
        if return_coeff:
            return coeff
        return design @ coeff - v.ravel()

    freq = least_squares(fit, [1.8, 2.4], bounds=([0.5, 0.8], [4.0, 4.0])).x
    return freq, fit(freq, return_coeff=True)


def _invert(x, freq, coeff):
    f, g = freq
    offset, gain, a, b = coeff[:4], coeff[4:8], coeff[8], coeff[9]
    shifted = x - offset
    z = shifted / gain
    # Solve the strictly increasing self-response exactly within each
    # paired update. The fitted cross-response is a contraction.
    for _ in range(240):
        rhs = shifted - b * np.sin(g * np.roll(z, 2, axis=1))
        q = rhs / gain
        for _ in range(8):
            t = np.tanh(f * q)
            q -= (gain * q + a * t - rhs) / (gain + a * f * (1 - t * t))
        delta = np.max(np.abs(q - z), initial=0.0)
        z = q
        if delta < 2e-12:
            break
    return z


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    a, b = study['panel_a'], study['panel_b']
    freq, coeff = _fit_sensor(np.asarray(a['u'], dtype=float), np.asarray(a['v'], dtype=float))
    operator = np.linalg.lstsq(_affine(np.asarray(b['u'], dtype=float)),
                              np.asarray(b['v'], dtype=float), rcond=None)[0]
    train = np.asarray(study['train']['x'], dtype=float)
    query = np.asarray(study['test']['x'], dtype=float)
    coords = _affine(_invert(np.concatenate((train, query)), freq, coeff)) @ operator
    return {'c': coords[:len(train)], 'query': coords[len(train):]}
