"""Reference-calibrated prediction with two nonlinear operators."""
import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _bias(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel, x):
    u, v = panel['u'], panel['v']
    def features(z, q):
        return np.stack((np.ones_like(z), z, np.tanh(q[0]*z),
                         np.sin(q[1]*np.roll(z, 2, axis=-1))), axis=-1)
    keep = np.linalg.norm(v, axis=1) < 4
    q = [1.8, 2.4]
    for _ in range(4):
        def coeff(q):
            a = features(u[keep], q)
            return np.stack([_ls(a[:, i], v[keep, i]) for i in range(4)])
        def residual(q):
            return ((features(u[keep], q)*coeff(q)).sum(-1)-v[keep]).ravel()
        q = least_squares(residual, q, bounds=([1.2, 2.0], [2.4, 2.8]), gtol=1e-12).x
        w = coeff(q)
        err = np.linalg.norm((features(u, q)*w).sum(-1)-v, axis=1)
        new = err <= max(.0015, 5*np.median(err))
        if np.array_equal(new, keep) or new.sum() < 12:
            break
        keep = new
    def invert(row):
        def residual(z):
            return (features(z, q)*w).sum(-1)-row
        return least_squares(residual, (row-w[:, 0])/(w[:, 1]+w[:, 2]), gtol=1e-12).x
    return np.stack([invert(row) for row in x]) if len(x) else np.empty((0, 4))


def _operators(panel, z):
    u = panel['u']
    def signed(x, q):
        return _bias(np.c_[np.tanh(q[0]*x), np.sin(q[1]*x)])
    def paired(x, q):
        p = x[:, [0, 0, 1, 2]]*x[:, [1, 3, 2, 3]]
        return _bias(np.c_[x, np.tanh(q[0]*x), p])
    def fit(fn, v, q, bounds):
        def residual(q):
            a = fn(u, q)
            return (a@_ls(a, v)-v).ravel()
        q = least_squares(residual, q, bounds=bounds, gtol=1e-12).x
        return fn(z, q)@_ls(fn(u, q), v)
    return (fit(signed, panel['v1'], [1.75, 2.1], ([1., 1.6], [2.5, 2.8])),
            fit(paired, panel['v2'], [1.2], (.6, 2.)))


def _design(study):
    train = study['train']
    z = _calibration(study['panel_a'], np.vstack((train['x'], study['test']['x'])))
    g, h = _operators(study['panel_c'], z)
    dot = np.einsum('ij,ij->i', g, h)
    scalar = _bias(np.c_[g, h, dot])
    fast = np.c_[g, dot]
    slow = np.c_[h, g[:, 0]*h[:, 1]]
    return scalar, fast, slow


def _trajectory(t, fast, slow, observed):
    n = len(observed)
    target = observed.ravel()

    def basis(rates):
        return -np.expm1(-t[:, None]*rates)

    def design(f, s, rates):
        b = basis(rates)
        # The second branch has both a slow component and a fast transient.
        return np.concatenate((f[:, None, :]*b[None, :, 1:2],
                               s[:, None, :]*b[None, :, :1],
                               s[:, None, :]*b[None, :, 1:2]), axis=2).reshape(-1, 15)

    def residual(rates):
        a = design(fast[:n], slow[:n], rates)
        return a@_ls(a, target)-target

    rates = least_squares(residual, [.4, 1.8], bounds=([.03, .9], [.89, 4.]),
                          gtol=1e-11).x
    coef = _ls(design(fast[:n], slow[:n], rates), target)
    return (design(fast[n:], slow[n:], rates)@coef).reshape(len(fast)-n, len(t))


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    scalar, fast, slow = _design(study)
    y = scalar[n:]@_ls(scalar[:n], train['y'])
    state = _trajectory(np.asarray(study['eval_times']), fast, slow, train['traj'])
    end = state[:, -1]
    return dict(y_hat=y, state_pred=state,
                diagnostics_vec=np.array([y.mean(), y.std(), end.mean(), end.std()]))
