"""Recover calibrated inputs and predict both outputs through panel B."""
import numpy as np
from scipy.optimize import least_squares


def _ls(f, y):
    return np.linalg.lstsq(f, y, rcond=None)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    r = np.linalg.norm(v - _aff(u) @ _ls(_aff(u), v), axis=1)
    keep = r < max(5*np.median(r), 2*np.quantile(r, .85))
    u, v = u[keep], v[keep]
    opp = (np.arange(u.shape[1])+2) % u.shape[1]

    def profile(p):
        h, k = np.tanh(p[0]*u), np.sin(p[1]*u[:, opp])
        a = np.stack((v-h*p[2]-k*p[3], u), axis=-1)
        a0, U = a[:, :, 0], a[:, :, 1]
        uc = U-U.mean(axis=0)
        slope = (uc*a0).sum(axis=0)/(uc**2).sum(axis=0)
        bias = np.mean(a0-U*slope, axis=0)
        return bias, slope, bias+U*slope+p[2]*h+p[3]*k-v

    p = least_squares(lambda p: profile(p)[2].ravel(), [1.8,2.4,.18,.32],
                      bounds=([.5,1.,0.,0.],[4.,4.,.7,.7])).x
    bias, slope, _ = profile(p)
    return p, bias, slope


def _inverse(x, calibration):
    p, bias, slope = calibration
    x = np.asarray(x, dtype=float)
    i = np.arange(x.shape[1]); j = (i+2) % len(i)
    z = (x-bias)/slope
    if len(x) == 0:
        return z
    jac = np.zeros((len(z),len(i),len(i)))
    for _ in range(35):
        h = np.tanh(p[0]*z)
        r = bias+slope*z+p[2]*h+p[3]*np.sin(p[1]*z[:,j])-x
        if np.max(abs(r)) < 1e-12:
            break
        jac[:,i,i] = slope+p[0]*p[2]*(1-h*h)
        jac[:,i,j] = p[1]*p[3]*np.cos(p[1]*z[:,j])
        z -= np.clip(np.linalg.solve(jac,r[...,None])[...,0], -.5, .5)
    return z


def _panel_b(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))

    def basis(z, frequency):
        return np.column_stack((_aff(z), np.cos(frequency * z)))

    def residual(p):
        f = basis(u, p[0])
        return (f @ _ls(f, v) - v).ravel()

    frequency = least_squares(residual, [2.6], bounds=([1.5], [3.5])).x[0]
    weights = _ls(basis(u, frequency), v)
    return lambda z: basis(z, frequency) @ weights


def _quadratic(z):
    i, j = np.triu_indices(z.shape[1])
    return np.column_stack((_aff(z), z[:, i] * z[:, j]))


def _trajectory(h, q, times, observed):
    a = _aff(h)
    projected = a @ _ls(a, observed)

    def basis(p):
        return -np.expm1(-times[:, None] * p)

    def residual(p):
        b = basis(p)
        return (b @ _ls(b, projected.T) - projected.T).ravel()

    rates = least_squares(residual, [.41, 1.8], bounds=([.05, .85], [.84, 5.])).x
    b = basis(rates)
    return (_aff(q) @ _ls(a, observed)) @ np.linalg.pinv(b.T) @ b.T


def solve_batch(study: dict) -> dict:
    train = study['train']
    calibration = _calibrate(study['panel_a'])
    panel_b = _panel_b(study['panel_b'])
    h = panel_b(_inverse(train['x'], calibration))
    q = panel_b(_inverse(study['test']['x'], calibration))
    y = _quadratic(q) @ _ls(_quadratic(h), np.asarray(train['y']))
    state = _trajectory(h, q, np.asarray(study['eval_times']), np.asarray(train['traj']))
    diagnostics = np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()]) if len(y) else np.zeros(4)
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diagnostics}
