"""Reference-panel calibration and two-mode trajectory regression."""
import numpy as np
from scipy.optimize import least_squares


def _ones(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _sensor_features(z, rates):
    return np.stack((np.ones_like(z), z, np.sin(rates[0] * np.roll(z, 2, axis=1)), np.tanh(rates[1] * z)), axis=2)


def _latent(panel, x):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    h = _ones(u)
    r = np.linalg.norm(h @ _fit(h, v) - v, axis=1)
    keep = r < 5 * max(np.median(r), 1e-7)
    if keep.sum() < 8:
        keep = np.ones(len(u), dtype=bool)

    def coefficients(h, mask):
        return np.stack([_fit(h[mask, j], v[mask, j]) for j in range(u.shape[1])])

    def project(h, c):
        return np.einsum('njk,jk->nj', h, c)

    def residual(rates):
        h = _sensor_features(u, rates)
        return (project(h, coefficients(h, keep)) - v)[keep].ravel()

    rates = least_squares(residual, [2.4, 1.8], bounds=([1.5, .4], [3.5, 4.]), max_nfev=80).x
    c = coefficients(_sensor_features(u, rates), keep)
    bias, scale, coupled, sat = c.T
    z = (x - bias) / scale
    j = np.arange(x.shape[1])
    for _ in range(30):
        residual_x = project(_sensor_features(z, rates), c) - x
        jac = np.zeros((len(x), len(j), len(j)))
        jac[:, j, j] = scale + sat * rates[1] * (1 - np.tanh(rates[1] * z)**2)
        jac[:, j, (j+2) % len(j)] = coupled * rates[0] * np.cos(rates[0] * np.roll(z, 2, axis=1))
        step = np.linalg.solve(jac, residual_x[..., None])[..., 0]
        z -= np.clip(step, -1., 1.)
        if np.max(np.abs(step), initial=0.) < 1e-11:
            break
    return z


def _edge(z):
    return z * np.roll(z, -1, axis=1)


def _design(panel, z):
    u = np.asarray(panel['u'])
    v1, v2 = np.asarray(panel['v1']), np.asarray(panel['v2'])
    c = _ones(_edge(z)) @ _fit(_ones(_edge(u)), v1)

    def residual(freq):
        h = _ones(np.cos(u * freq[0]))
        return (h @ _fit(h, v2) - v2).ravel()

    f = least_squares(residual, [2.4], bounds=([1.5], [3.5])).x[0]
    d = _ones(np.cos(f * z)) @ _fit(_ones(np.cos(f * u)), v2)
    h = _ones(np.column_stack((c, d, np.sum(c * d, axis=1))))
    return h, np.column_stack((h, c[:, 0] * d[:, 1]))


def _predict(study):
    tr = study['train']
    n = len(tr['x'])
    x = np.concatenate((tr['x'], study['test']['x']), axis=0)
    h, g = _design(study['panel_c'], _latent(study['panel_a'], x))
    y = h[n:] @ _fit(h[:n], tr['y'])
    t = np.asarray(study['eval_times'])
    traj = np.asarray(tr['traj'])
    weights = _fit(g[:n], traj)
    projected = (g[:n] @ weights).T

    def modes(rates):
        return -np.expm1(-t[:, None] * rates)

    def residual(rates):
        q = modes(rates)
        return (q @ _fit(q, projected) - projected).ravel()

    rates = least_squares(residual, [.42, 1.82], bounds=([.05, .85], [.85, 4.]), max_nfev=100).x
    q, _ = np.linalg.qr(modes(rates), mode='reduced')
    state = (g[n:] @ (weights @ q)) @ q.T
    return y, state


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    sparse = inputs['node_00']
    y, state = _predict(study)
    # Express the calibrated estimate relative to the frozen sparse baseline,
    # in the panel coordinates used by the fixed downstream blend.
    baseline_y, baseline_state = sparse['y_hat'], sparse['state_pred']
    return {'panel_pred': np.column_stack((baseline_y + (y - baseline_y) / .5,
                                          baseline_state + (state - baseline_state) / .65))}
