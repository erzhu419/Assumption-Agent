"""Recover the calibrated dual-operator graph and its two-mode response."""
import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _with_bias(a):
    return np.column_stack((np.ones(len(a)), a))


def _calibrate(panel, raw):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    k = u.shape[1]
    j = (np.arange(k) + 1) % k

    def terms(a):
        b = a[..., j]
        return np.stack((np.ones_like(a), a, b, a*b, a**3), axis=-1)

    f = terms(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(10):
        w = np.stack([_ls(f[keep, i], v[keep, i]) for i in range(k)])
        residual = np.linalg.norm(np.sum(f*w, axis=-1)-v, axis=1)
        refined = residual <= max(5*np.median(residual), 0.0008)
        if np.array_equal(refined, keep) or refined.sum() <= f.shape[-1]:
            break
        keep = refined

    rows = np.arange(k)
    linear = np.diag(w[:, 1])
    linear[rows, j] = w[:, 2]
    initial = np.linalg.solve(linear, (raw - w[:, 0]).T).T

    def jac(a):
        out = np.diag(w[:, 1] + w[:, 3]*a[j] + 3*w[:, 4]*a*a)
        out[rows, j] = w[:, 2] + w[:, 3]*a
        return out

    def invert(row, start):
        def residual(a):
            return np.sum(terms(a)*w, axis=-1) - row
        return least_squares(residual, start, jac=jac, gtol=1e-12).x

    return np.stack([invert(row, start) for row, start in zip(raw, initial)])


def _operators(panel, z):
    u = np.asarray(panel['u'])

    def products(a):
        return _with_bias(a[:, [0, 0, 1, 2]] * a[:, [1, 3, 2, 3]])

    def cosine(a, q):
        return _with_bias(np.cos(q*a))

    def residual(q):
        f = cosine(u, q[0])
        return (f @ _ls(f, panel['v2']) - panel['v2']).ravel()

    q = least_squares(residual, [2.4], bounds=(1.5, 3.3), gtol=1e-12).x[0]
    g = products(z) @ _ls(products(u), panel['v1'])
    h = cosine(z, q) @ _ls(cosine(u, q), panel['v2'])
    return g, h


def _temporal(t, traj):
    def basis(rates):
        return -np.expm1(-t[:, None]*rates)

    def residual(rates):
        b = basis(rates)
        return (b @ _ls(b, traj.T) - traj.T).ravel()

    rates = least_squares(residual, [0.42, 1.8], bounds=([0.03, 0.8], [0.8, 5.0]),
                          gtol=1e-12).x
    return basis(rates)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.vstack((train['x'], study['test']['x']))
    z = _calibrate(study['panel_a'], x)
    g, h = _operators(study['panel_c'], z)
    y_design = _with_bias(np.column_stack((g, h, np.sum(g*h, axis=1))))
    interaction = (g[:, :, None]*h[:, None, :]).reshape(len(x), -1)
    state_design = _with_bias(np.column_stack((g, h, interaction)))
    time_design = _temporal(np.asarray(study['eval_times']), train['traj'])
    amplitudes = _ls(time_design, train['traj'].T).T
    y_hat = y_design[n:] @ _ls(y_design[:n], train['y'])
    state_pred = (state_design[n:] @ _ls(state_design[:n], amplitudes)) @ time_design.T
    diagnostics_vec = np.array([y_hat.mean(), y_hat.std(), state_pred[:, -1].mean(),
                                state_pred[:, -1].std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics_vec}
