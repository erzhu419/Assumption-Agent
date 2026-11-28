import numpy as np
from scipy.optimize import least_squares, root


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _calibrate(panel, frequency):
    u, v = panel['u'], panel['v']
    magnitude = np.max(np.abs(v), axis=1)
    keep = magnitude <= 2.5 * np.median(magnitude)
    if np.count_nonzero(keep) < 8:
        keep = np.ones(len(u), dtype=bool)
    n_dim = u.shape[1]
    partner = (np.arange(n_dim) + 2) % n_dim

    def fit(mask):
        x, y = u[mask], v[mask]
        base = np.stack((np.ones_like(x), x), axis=-1).reshape(-1, 2)
        channels = np.tile(np.eye(n_dim), (len(x), 1))
        diagonal = (base[:, :, None] * channels[:, None, :]).reshape(len(base), -1)

        def design(freqs):
            return np.column_stack((diagonal, np.sin(freqs[0] * x).ravel(), np.sin(freqs[1] * x[:, partner]).ravel()))

        def residual(freqs):
            d = design(freqs)
            return d @ _ls(d, y.ravel()) - y.ravel()

        start = np.clip([frequency + 0.9, frequency + 0.25], 0.6, 4.9)
        freqs = least_squares(residual, start, bounds=(0.5, 5.), gtol=1e-11).x
        coef = _ls(design(freqs), y.ravel())
        return coef[:n_dim], coef[n_dim:2*n_dim], coef[-2:], freqs, partner

    for _ in range(4):
        pars = fit(keep)
        residual = np.linalg.norm(_forward(u, pars) - v, axis=1)
        new_keep = residual <= max(0.015, 5 * np.median(residual))
        if np.count_nonzero(new_keep) < 8 or np.array_equal(keep, new_keep):
            return pars
        keep = new_keep
    return fit(keep)


def _forward(u, pars):
    offset, slope, amp, freq, partner = pars
    return offset + slope*u + amp[0]*np.sin(freq[0]*u) + amp[1]*np.sin(freq[1]*u[..., partner])


def _invert(x, pars):
    offset, slope, amp, freq, partner = pars
    initial = (x - offset) / slope
    z = initial.copy()
    for _ in range(32):
        residual = _forward(z, pars) - x
        if not residual.size or np.max(np.abs(residual)) < 1e-10:
            return z
        z -= residual / slope

    def jac(t):
        j = np.diag(slope + amp[0]*freq[0]*np.cos(freq[0]*t))
        j[np.arange(len(t)), partner] += amp[1]*freq[1]*np.cos(freq[1]*t[partner])
        return j

    for i in np.flatnonzero(np.max(np.abs(_forward(z, pars)-x), axis=1) >= 1e-10):
        def fun(t):
            return _forward(t, pars) - x[i]
        fit = root(fun, z[i], jac=jac, tol=1e-10)
        if np.max(np.abs(fit.fun)) > 1e-8:
            fit = least_squares(fun, fit.x, jac=jac, gtol=1e-12)
        z[i] = fit.x
    return z


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x = np.concatenate((study['train']['x'], study['test']['x']), axis=0)
    params = _calibrate(study['panel_a'], float(study['telemetry_vec'][0]))
    u = _invert(x, params)
    b = study['panel_b']
    return {'w': _aff(u) @ _ls(_aff(b['u']), b['v'])}
