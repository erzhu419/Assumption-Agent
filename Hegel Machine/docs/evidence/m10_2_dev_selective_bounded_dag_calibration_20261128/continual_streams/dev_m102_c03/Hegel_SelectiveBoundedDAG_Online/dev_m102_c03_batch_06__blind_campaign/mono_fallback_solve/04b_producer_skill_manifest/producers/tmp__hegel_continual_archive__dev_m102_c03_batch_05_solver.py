"""Calibrated dual-branch regression with shared exponential dynamics."""
import numpy as np
from scipy.optimize import least_squares


def _fit(h, targets):
    return np.linalg.lstsq(h, targets, rcond=None)[0]


def _one(x):
    return np.column_stack((np.ones(len(x)), x))


def _edge(x):
    return x * np.roll(x, -1, axis=1)


def _cal_design(x, j):
    u, v = x[:, j], x[:, (j + 1) % 4]
    return _one(np.column_stack((u, v, u*v, u**3)))


def _calibration(panel):
    u, v = panel['u'], panel['v']
    h = _one(u)
    r = np.linalg.norm(h @ _fit(h, v) - v, axis=1)
    good = r <= 4 * max(np.median(r), 1e-8)
    for _ in range(4):
        a = np.stack([_fit(_cal_design(u[good], j), v[good, j]) for j in range(4)])
        r = np.linalg.norm(np.column_stack([_cal_design(u, j) @ a[j] for j in range(4)]) - v, axis=1)
        new = r <= 5 * max(np.median(r), 1e-6)
        if new.sum() < 10 or np.array_equal(good, new):
            break
        good = new
    return a


def _inverse(x, a):
    bias, scale, neighbor, mixed, cubic = a.T
    z = (x - bias) / scale
    j = np.arange(4)
    for _ in range(14):
        w = np.roll(z, -1, axis=1)
        r = bias + scale*z + neighbor*w + mixed*z*w + cubic*z**3 - x
        jac = np.zeros((len(x), 4, 4))
        jac[:, j, j] = scale + mixed*w + 3*cubic*z*z
        jac[:, j, (j + 1) % 4] = neighbor + mixed*z
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        z -= np.clip(step, -1., 1.)
        if np.max(np.abs(step), initial=0.) < 1e-12:
            break
    return z


def _branches(study, x):
    z = _inverse(np.asarray(x, dtype=float), _calibration(study['panel_a']))
    u, v1, v2 = (study['panel_c'][k] for k in ('u', 'v1', 'v2'))
    c = _one(_edge(z)) @ _fit(_one(_edge(u)), v1)

    def cosine_residual(f):
        h = _one(np.cos(u * f[0]))
        return (h @ _fit(h, v2) - v2).ravel()

    f = least_squares(cosine_residual, [2.4], bounds=(0.1, 7.)).x[0]
    d = _one(np.cos(f*z)) @ _fit(_one(np.cos(f*u)), v2)
    return c, d


def _fit_state(h, t, traj):
    # Variable projection: fit linear spatial coefficients at each trial rate pair.
    pinv = np.linalg.pinv(h)
    proj = h @ (pinv @ traj)

    def modes(g):
        return -np.expm1(-g[:, None] * t)

    def residual(g):
        q = modes(g)
        return (_fit(q.T, proj.T).T @ q - proj).ravel()

    rates = least_squares(residual, [.42, 1.82], bounds=([.05, .8], [.8, 4.]), ftol=1e-11, gtol=1e-11).x
    q = modes(rates)
    return pinv @ _fit(q.T, traj.T).T, q


def solve_batch(study):
    train = study['train']
    n = len(train['x'])
    c, d = _branches(study, np.concatenate((train['x'], study['test']['x']), axis=0))
    dot = np.sum(c*d, axis=1)
    hy = _one(np.column_stack((c, d, dot)))
    ht = np.column_stack((hy, c[:, 0]*d[:, 1]))
    y_hat = hy[n:] @ _fit(hy[:n], train['y'])
    weights, modes = _fit_state(ht[:n], np.asarray(study['eval_times']), train['traj'])
    state = ht[n:] @ weights @ modes
    diagnostics = np.array([y_hat.mean(), y_hat.std(), state[:, -1].mean(), state[:, -1].std()])
    return {'y_hat': y_hat, 'state_pred': state, 'diagnostics_vec': diagnostics}
