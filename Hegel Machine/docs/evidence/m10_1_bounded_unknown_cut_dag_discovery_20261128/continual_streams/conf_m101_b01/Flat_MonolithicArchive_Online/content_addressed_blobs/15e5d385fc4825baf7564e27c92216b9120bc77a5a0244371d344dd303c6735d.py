"""Infer the calibration, cyclic readout, and shared trajectory modes."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import erf


def _fit(d, y):
    return np.linalg.lstsq(d, y, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _cal_design(x):
    v = np.roll(x, -1, axis=1)
    return np.stack((np.ones_like(x), x, v, x*v, x**3), axis=-1)


def _calibration(panel):
    u, v = panel['u'], panel['v']
    d = _cal_design(u)
    good = np.ones(len(u), dtype=bool)
    for _ in range(8):
        w = np.stack([_fit(d[good, j], v[good, j]) for j in range(u.shape[1])])
        e = np.linalg.norm(np.einsum('nki,ki->nk', d, w)-v, axis=1)
        good = e <= np.sort(e)[int(.79*len(u))]
    good = e < 12*np.median(e)
    return np.stack([_fit(d[good, j], v[good, j]) for j in range(u.shape[1])])


def _inverse_calibration(measured, w):
    x = (measured-w[:, 0])/w[:, 1]
    j = np.arange(x.shape[1])
    for _ in range(18):
        nxt = np.roll(x, -1, axis=1)
        jac = np.zeros((len(x), len(j), len(j)))
        jac[:, j, j] = w[:, 1]+w[:, 3]*nxt+3*w[:, 4]*x*x
        jac[:, j, (j+1)%len(j)] = w[:, 2]+w[:, 3]*x
        res = np.einsum('nki,ki->nk', _cal_design(x), w)-measured
        step = np.linalg.solve(jac, res[..., None])[..., 0]
        x -= step
        if np.max(np.abs(step), initial=0.) < 1e-11:
            break
    return x


def _operator_b(panel, x):
    u, v = panel['u'], panel['v']

    def design(z, k):
        return np.column_stack((_affine(z), erf(k*z)))

    def loss(k):
        d = design(u, k)
        return np.mean((d@_fit(d, v)-v)**2)

    k = minimize_scalar(loss, bounds=(.6, 2.8), method='bounded').x
    return design(x, k)@_fit(design(u, k), v)


def _cycle_features(panel, x):
    i, j = np.triu_indices(x.shape[1], 1)
    u = panel['u']
    w = _fit(_affine(u[:, i]*u[:, j]), panel['v1'])[1:]
    strength = np.linalg.norm(w, axis=1)
    keep = strength > .02*strength.max()
    return _affine(x[:, i[keep]]*x[:, j[keep]])


def _modes(times, log_rates):
    return -np.expm1(-times[:, None]*np.exp(log_rates)[None, :])


def _trajectory(d, dq, times, z):
    def residual(p):
        h = _modes(times, p)
        return (h@_fit(h, z.T)-z.T).ravel()

    rates = least_squares(residual, [-1., .6], bounds=(-7., 4.)).x
    h = _modes(times, rates)
    # Preserve the measured common offset; only the spatial variations use the two modes.
    avg = z.mean(axis=0)
    a = _fit(h, (z-avg).T).T
    return avg+(dq@_fit(d, a))@h.T


def solve_batch(study: dict) -> dict:
    tr = study['train']
    n = len(tr['x'])
    x = np.concatenate((tr['x'], study['test']['x']), axis=0)
    x = _inverse_calibration(x, _calibration(study['panel_a']))
    d = _cycle_features(study['panel_c'], _operator_b(study['panel_b'], x))
    y = d[n:]@_fit(d[:n], tr['y'])
    z = _trajectory(d[:n], d[n:], study['eval_times'], tr['traj'])
    return {'y_hat': y, 'state_pred': z,
            'diagnostics_vec': np.array([y.mean(), y.std(), z[:, -1].mean(), z[:, -1].std()])}
