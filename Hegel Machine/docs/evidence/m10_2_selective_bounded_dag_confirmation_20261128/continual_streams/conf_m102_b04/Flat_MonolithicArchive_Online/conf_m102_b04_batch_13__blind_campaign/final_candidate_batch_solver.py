"""Calibrate the opposing-channel sensor, then fit cosine-operator readouts."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    distance = np.linalg.norm(v-u, axis=1)
    med = np.median(distance)
    keep = distance < med + 8*np.median(abs(distance-med))
    if keep.sum() < 20:
        keep[:] = True
    eye = np.broadcast_to(np.eye(4), (len(u), 4, 4))
    affine = np.concatenate((eye, eye*u[..., None]), axis=-1)
    opposite = np.roll(u, -2, axis=1)

    def design(p):
        return np.concatenate((affine, np.tanh(p[0]*u)[..., None],
                               np.sin(p[1]*opposite)[..., None]), axis=-1)

    for _ in range(3):
        def residual(p):
            d = design(p)[keep].reshape(-1, 10)
            obs = v[keep].ravel()
            return d@_ls(d, obs)-obs
        p = least_squares(residual, [1.8, 2.4], bounds=([.5, 1.4], [4., 3.5]),
                          gtol=1e-13).x
        d = design(p)[keep].reshape(-1, 10)
        w = _ls(d, v[keep].ravel())
        err = np.linalg.norm(design(p)@w-v, axis=1)
        mid = np.median(err)
        new_keep = err <= mid+8*max(np.median(abs(err-mid)), 1e-9)
        if np.array_equal(keep, new_keep) or new_keep.sum() < 20:
            break
        keep = new_keep
    return p, w


def _inverse(x, calibration):
    (k, f), w = calibration
    bias, gain, a, b = w[:4], w[4:8], w[8], w[9]
    z = (x-bias)/gain
    i = np.arange(4)
    jac = np.zeros((len(x), 4, 4))
    for _ in range(50):
        opp = np.roll(z, -2, axis=1)
        h = np.tanh(k*z)
        r = bias+gain*z+a*h+b*np.sin(f*opp)-x
        if np.max(abs(r), initial=0.) < 1e-12:
            break
        jac[:, i, i] = gain+a*k*(1-h*h)
        jac[:, i, (i+2)%4] = b*f*np.cos(f*opp)
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        z -= step/np.maximum(1., np.max(abs(step), axis=1, keepdims=True))
    return z


def _operator(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    def loss(f):
        d = _affine(np.cos(f*u))
        return np.sum((d@_ls(d, v)-v)**2)
    grid = np.linspace(.2, 6., 117)
    j = np.argmin([loss(f) for f in grid])
    f = minimize_scalar(loss, bounds=(grid[max(0,j-1)], grid[min(len(grid)-1,j+1)]),
                        method='bounded', options={'xatol': 1e-10}).x
    w = _ls(_affine(np.cos(f*u)), v)
    return lambda z: _affine(np.cos(f*z))@w


def _scalar_design(h):
    return np.column_stack((_affine(h), h[:, 0]*h[:, 2]+h[:, 1]*h[:, 3]))


def _trajectory(h, q, times, trajectory):
    a, b = _affine(h), _affine(q)
    beta = _ls(a, trajectory)
    fitted = a@beta
    def basis(r):
        return -np.expm1(-times[:, None]*r)
    def residual(r):
        f = basis(r)
        return (f@_ls(f, fitted.T)-fitted.T).ravel()
    rates = least_squares(residual, [.4, 1.8], bounds=([.03, .9], [1., 6.]), gtol=1e-11).x
    f = basis(rates)
    return b@_ls(f, beta.T).T@f.T


def solve_batch(study: dict) -> dict:
    train = study['train']
    calibration = _calibration(study['panel_a'])
    operator = _operator(study['panel_b'])
    h = operator(_inverse(np.asarray(train['x'], dtype=float), calibration))
    q = operator(_inverse(np.asarray(study['test']['x'], dtype=float), calibration))
    y = _scalar_design(q)@_ls(_scalar_design(h), train['y'])
    state = _trajectory(h, q, np.asarray(study['eval_times'], dtype=float), train['traj'])
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
