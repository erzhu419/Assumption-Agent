"""Calibrate the observation map and panel channels for the fixed DAG sink."""
import numpy as np
from scipy.optimize import least_squares


def _bias(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-10)[0]


def _calibrate(u, v, design, start, bounds):
    def residual(p):
        f = design(u, p)
        return (f @ _ls(f, v) - v).ravel()
    p = least_squares(residual, start, bounds=bounds, max_nfev=100).x
    return p, _ls(design(u, p), v)


def _coordinates(panel, x):
    u, v = np.asarray(panel['u'], float), np.asarray(panel['v'], float)
    dist = np.linalg.norm(v - u, axis=1)
    good = dist <= max(1e-8, 4 * np.median(dist))

    def design(z, p):
        return _bias(np.column_stack((z, np.sin(p[0] * z), np.tanh(p[1] * z))))

    p, coef = _calibrate(u[good], v[good], design, [2.4, 1.75], ([1.8, .8], [3.1, 3.1]))
    z = x.copy()
    for _ in range(30):
        r = design(z, p) @ coef - x
        if not r.size or np.max(np.abs(r)) < 1e-10:
            break
        ds = p[0] * np.cos(p[0] * z)
        dt = p[1] * (1 - np.tanh(p[1] * z)**2)
        jac = coef[1:5].T + ds[:, None, :] * coef[5:9].T + dt[:, None, :] * coef[9:13].T
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        z -= np.clip(step, -.35, .35)
    return z


def _features(panel, z):
    u = np.asarray(panel['u'], float)

    def first(q, p):
        w = p[1] * q
        return _bias(np.column_stack((np.tanh(p[0] * q), w / (1 + np.abs(w)))))

    def second(q, p):
        return _bias(np.column_stack((q, np.expm1(-p[0] * q*q))))

    p, coef = _calibrate(u, np.asarray(panel['v1'], float), first, [1.53, 1.5], ([.4, .4], [3.5, 3.5]))
    a = first(z, p) @ coef
    p, coef = _calibrate(u, np.asarray(panel['v2'], float), second, [.29], ([.05], [1.2]))
    b = second(z, p) @ coef
    return _bias(np.column_stack((a, b, np.sum(a*b, axis=1), a[:, 0]*b[:, 1])))


def run_node(inputs: dict) -> dict:
    study, sparse = inputs['__input__'], inputs['node_00']
    train = study['train']
    x = np.asarray(train['x'], float)
    z = _coordinates(study['panel_a'], np.vstack((x, study['test']['x'])))
    f = _features(study['panel_c'], z)
    fit, pred = f[:len(x)], f[len(x):]
    y = pred[:, :-1] @ _ls(fit[:, :-1], np.asarray(train['y'], float))
    traj = np.asarray(train['traj'], float)
    basis = np.linalg.svd(traj, full_matrices=False)[2][:2]
    state = (pred @ _ls(fit, traj @ basis.T)) @ basis
    # Encode the calibrated estimate as a correction relative to node_00.
    # The untouched sink applies different correction weights to y and state.
    y0, state0 = np.asarray(sparse['y_hat']), np.asarray(sparse['state_pred'])
    return {'panel_pred': np.column_stack((y0 + (y-y0)/.5, state0 + (state-state0)/.65))}
