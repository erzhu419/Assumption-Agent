"""Reconstruct the calibrated coordinates and predict the two readouts."""
import numpy as np
from scipy.optimize import least_squares, root


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _cal_basis(z):
    q = np.roll(z, -1, axis=-1)
    return np.stack((np.ones_like(z), z, q, z*q, z**3), axis=-1)


def _cal(panel, x):
    d = _cal_basis(panel['u'])
    co = []
    for j in range(d.shape[1]):
        a, b = d[:, j], panel['v'][:, j]
        ix = np.arange(len(a))
        for _ in range(12):
            res = abs(a @ _ls(a[ix], b[ix]) - b)
            ix = np.argsort(res)[:int(.84*len(a))]
        ix = res < max(.0007, 8*np.median(res))
        co.append(_ls(a[ix], b[ix]))
    co = np.asarray(co)

    def inverse(row):
        def residual(z):
            return (_cal_basis(z)*co).sum(axis=-1) - row
        opt = root(residual, row, tol=1e-10)
        if np.max(abs(opt.fun)) < 1e-8:
            return opt.x
        return least_squares(residual, opt.x, gtol=1e-12).x

    return np.array([inverse(row) for row in x]).reshape(-1, 4)


def _dead(z):
    return _aff(np.sign(z)*np.maximum(abs(z)-.35, 0)**1.5)


def _c_basis(x, rates):
    a, b = rates
    return _aff(np.column_stack((x, np.tanh(a*x), np.sin(b*x))))


def _c_fit(c, telemetry):
    u, v = c['u'], c['v1']
    def residual(rates):
        d = _c_basis(u, rates)
        return (d @ _ls(d, v) - v).ravel()
    rates = least_squares(residual, [1.75, telemetry[3]], bounds=([.65, 1.4], [3., 3.2]), ftol=1e-12, gtol=1e-12, xtol=1e-12).x
    return rates, _ls(_c_basis(u, rates), v)


def _coords(study, x):
    z = _cal(study['panel_a'], x)
    b = study['panel_b']
    w = _dead(z) @ _ls(_dead(b['u']), b['v'])
    rates, co = _c_fit(study['panel_c'], study['telemetry_vec'])
    return w, _c_basis(w, rates) @ co


def _interactions(q):
    return np.column_stack((q[:,0]*q[:,2], q[:,1]*q[:,3]))


def _time_basis(t, rates):
    return -np.expm1(-np.asarray(t)[:, None]*rates)


def _trajectory(d, dt, t, traj):
    def residual(rates):
        b = _time_basis(t, rates)
        return (b @ _ls(b, traj.T) - traj.T).ravel()
    rates = least_squares(residual, [1.8, .4], bounds=([1., .05], [4., .99])).x
    b = _time_basis(t, rates)
    return dt @ _ls(d, _ls(b, traj.T).T) @ b.T


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    w, q = _coords(study, np.concatenate((train['x'], study['test']['x'])))
    dy = _aff(np.column_stack((q, w, _interactions(q))))
    dp = _aff(np.column_stack((q, w)))
    y = dy[n:] @ _ls(dy[:n], train['y'])
    p = _trajectory(dp[:n], dp[n:], study['eval_times'], train['traj'])
    last = p[:, -1]
    return {'y_hat': y, 'state_pred': p,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
