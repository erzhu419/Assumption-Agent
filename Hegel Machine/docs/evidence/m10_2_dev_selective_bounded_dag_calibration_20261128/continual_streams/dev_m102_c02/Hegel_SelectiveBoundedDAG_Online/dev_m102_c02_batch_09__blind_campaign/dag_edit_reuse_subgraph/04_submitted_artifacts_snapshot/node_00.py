import numpy as np
from scipy.optimize import least_squares


def _calibrate_a(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    # Gross row corruptions are separated from ordinary channel coupling.
    distance = np.linalg.norm(v - u, axis=1)
    center = np.median(distance)
    mad = np.median(np.abs(distance - center))
    keep = distance <= center + max(1.0, 8.0 * mad)
    u, v = u[keep], v[keep]
    n, d = u.shape
    fixed = np.zeros((n, d, 2*d))
    idx = np.arange(d)
    fixed[:, idx, idx] = 1.
    fixed[:, idx, d + idx] = u
    opposite = np.roll(u, 2, axis=1)

    def profile(freq):
        design = np.column_stack((fixed.reshape(n*d, 2*d),
                                  np.tanh(freq[0] * u).ravel(),
                                  np.sin(freq[1] * opposite).ravel()))
        coeff = np.linalg.lstsq(design, v.ravel(), rcond=None)[0]
        return design @ coeff - v.ravel(), coeff

    opt = least_squares(lambda p: profile(p)[0], [1.8, 2.4],
                        bounds=([0.3, 0.5], [5., 5.]), max_nfev=120)
    coeff = profile(opt.x)[1]
    return coeff[:d], coeff[d:2*d], coeff[-2], coeff[-1], *opt.x


def _invert_a(x, model):
    x = np.asarray(x, dtype=float)
    bias, gain, a, b, k, w = model
    d = len(gain)
    idx = np.arange(d)
    z = (x - bias) / gain
    def residual(p, y):
        return bias + gain*p + a*np.tanh(k*p) + b*np.sin(w*np.roll(p, 2)) - y
    def jacobian(p):
        j = np.diag(gain + a*k*(1. - np.tanh(k*p)**2))
        j[idx, (idx + 2) % d] += b*w*np.cos(w*np.roll(p, 2))
        return j
    for i, y in enumerate(x):
        z[i] = least_squares(lambda p: residual(p, y), z[i], jac=jacobian, gtol=1e-12, ftol=1e-12, xtol=1e-12, max_nfev=100).x
    return z


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    sensor = _calibrate_a(study['panel_a'])
    panel = study['panel_b']
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    operator = np.linalg.lstsq(_affine(u), v, rcond=None)[0]

    def coordinates(x):
        return _affine(_invert_a(x, sensor)) @ operator

    return {'c': coordinates(study['train']['x']),
            'query': coordinates(study['test']['x'])}
