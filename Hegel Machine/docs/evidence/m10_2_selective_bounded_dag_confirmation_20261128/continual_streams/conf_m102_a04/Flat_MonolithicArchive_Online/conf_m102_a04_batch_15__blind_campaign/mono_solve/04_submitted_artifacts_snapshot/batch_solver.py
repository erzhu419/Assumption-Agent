"""Panel-calibrated bilinear predictor with a shared two-mode time response."""
import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _one(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel, raw):
    u, v = panel['u'], panel['v']

    def features(z):
        b = np.roll(z, -1, axis=-1)
        return np.stack((np.ones_like(z), z, b, z*b, z**3), axis=-1)

    f = features(u)
    keep = np.linalg.norm(v, axis=1) < 5
    for _ in range(3):
        w = np.stack([_ls(f[keep, j], v[keep, j]) for j in range(4)])
        r = np.linalg.norm(np.sum(f*w, axis=-1)-v, axis=1)
        keep = r < max(0.001, 6*np.median(r))
    w = np.stack([_ls(f[keep, j], v[keep, j]) for j in range(4)])

    def invert(row):
        def residual(z):
            return np.sum(features(z)*w, axis=-1)-row
        return least_squares(residual, (row-w[:, 0])/w[:, 1], gtol=1e-11).x

    return np.stack([invert(row) for row in raw])


def _profile(design, u, v, start, bounds):
    def residual(q):
        a = design(u, q)
        return (a@_ls(a, v)-v).ravel()
    q = least_squares(residual, start, bounds=bounds, gtol=1e-11, max_nfev=250).x
    return q, _ls(design(u, q), v)


def _operators(panel, z):
    u = panel['u']

    def signed(x, q):
        return _one(np.c_[np.tanh(q[0]*x), np.sin(q[1]*x), x/(1+q[2]*np.abs(x))])

    def curved(x, q):
        return _one(np.c_[x, np.tanh(q[0]*x), x*np.roll(x, -1, axis=1)])

    q, a = _profile(signed, u, panel['v1'], [2., 2., 1.], ([.3,.3,.1], [6.,6.,6.]))
    r, b = _profile(curved, u, panel['v2'], [1.2], (.2, 4.))
    return signed(z, q)@a, curved(z, r)@b


def _time(t, r):
    return -np.expm1(-t[:, None]*r)


def _readout(g, h):
    dot = np.sum(g*h, axis=1, keepdims=True)
    cross = (g[:, :, None]*h[:, None, :]).reshape(len(g), 16)
    return _one(np.c_[g, h, dot]), _one(np.c_[g, h, cross])


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    raw = np.vstack((train['x'], study['test']['x']))
    z = _calibrate(study['panel_a'], raw)
    g, h = _operators(study['panel_c'], z)
    scalar, spatial = _readout(g, h)
    y = scalar[n:] @ _ls(scalar[:n], train['y'])
    target = np.asarray(train['traj']).T
    t = np.asarray(study['eval_times'])

    def residual(r):
        f = _time(t, r)
        return (f@_ls(f, target)-target).ravel()

    r = least_squares(residual, [.42, 1.82], bounds=([.05,.8], [.79,4.]), gtol=1e-11).x
    time = _time(t, r)
    amplitude = _ls(time, target).T
    state = (spatial[n:] @ _ls(spatial[:n], amplitude)) @ time.T
    end = state[:, -1]
    return dict(y_hat=y, state_pred=state,
                diagnostics_vec=np.array([y.mean(), y.std(), end.mean(), end.std()]))
