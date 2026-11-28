"""Calibrated dual-operator prediction with separate scalar and dynamic readouts."""
import numpy as np
from scipy.optimize import least_squares


def _linear(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _intercept(x):
    return np.column_stack((np.ones(len(x)), x))


def _profile(basis, u, v, start, bounds):
    def residual(q):
        a = basis(u, q)
        return (a @ _linear(a, v) - v).ravel()
    q = least_squares(residual, start, bounds=bounds, gtol=1e-12).x
    a = basis(u, q)
    return q, _linear(a, v)


def _calibrate(panel, raw):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    norms = np.linalg.norm(v, axis=1)
    keep = norms < 3 * np.median(norms)
    u, v = u[keep], v[keep]
    opposite = [2, 3, 0, 1]

    def basis(z, q):
        return np.stack((np.ones_like(z), z, np.sin(q[0]*z),
                         np.sin(q[1]*z[:, opposite])), axis=2)

    def fit(q):
        a = basis(u, q)
        w = np.stack([_linear(a[:, i], v[:, i]) for i in range(v.shape[1])])
        return w, (a*w).sum(axis=2)-v

    q = least_squares(lambda q: fit(q)[1].ravel(), [3.1, 2.4],
                      bounds=([1, 1], [4, 4]), gtol=1e-12).x
    w = fit(q)[0]
    z = []
    for row in raw:
        def residual(x):
            return (basis(x[None, :], q)[0]*w).sum(axis=1)-row
        z.append(least_squares(residual, (row-w[:, 0])/w[:, 1], gtol=1e-12).x)
    return np.asarray(z)


def _operators(panel, z):
    u = np.asarray(panel['u'])

    def signed(x, q):
        return _intercept(x/(1+q[0]*np.abs(x))+q[1]*x)

    def curved(x, q):
        x2 = x*x
        return np.column_stack((_intercept(x), x2, x2/(1+q[0]*x2)))

    q, w = _profile(signed, u, panel['v1'], [.83, -.16], ([.1, -.5], [4, .5]))
    r, v = _profile(curved, u, panel['v2'], [.29], (.1, .5))
    return signed(z, q) @ w, curved(z, r) @ v


def _time(t, q):
    return -np.expm1(-t[:, None]*q)


def _tensor_readout(f0, f1, t, traj, n):
    def design(q, split):
        a = _time(t, q)
        return np.concatenate([(f[split, None, :]*a[None, :, k, None])
                               for k, f in enumerate((f0, f1))], axis=2)

    def profiled(q):
        a = design(q, slice(0, n)); a = a.reshape(-1, a.shape[-1])
        w = _linear(a, traj.ravel())
        return w, a @ w - traj.ravel()

    q = least_squares(lambda q: profiled(q)[1], [.42, 1.81],
                      bounds=([.12, .9], [.8, 3.5]), gtol=1e-11).x
    return design(q, slice(n, None)) @ profiled(q)[0]


def solve_batch(study: dict) -> dict:
    train, test = study['train'], study['test']
    n = len(train['x'])
    z = _calibrate(study['panel_a'], np.vstack((train['x'], test['x'])))
    g, h = _operators(study['panel_c'], z)
    dot = np.sum(g*h, axis=1)
    cross = g[:, 0]*h[:, 1]
    f = _intercept(np.column_stack((g, h, dot)))
    y = f[n:] @ _linear(f[:n], train['y'])
    f0 = _intercept(np.column_stack((g, h, cross)))
    f1 = _intercept(np.column_stack((g, h, dot)))
    state = _tensor_readout(f0, f1, np.asarray(study['eval_times']), np.asarray(train['traj']), n)
    end = state[:, -1]
    return dict(y_hat=y, state_pred=state,
                diagnostics_vec=np.array([y.mean(), y.std(), end.mean(), end.std()]))
