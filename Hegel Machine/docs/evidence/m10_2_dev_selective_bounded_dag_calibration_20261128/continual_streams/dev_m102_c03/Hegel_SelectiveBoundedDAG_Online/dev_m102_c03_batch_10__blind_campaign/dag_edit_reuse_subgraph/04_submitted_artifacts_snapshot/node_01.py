"""Calibrate the reference panels and express the readout in frozen sink coordinates."""
import numpy as np
from scipy.optimize import minimize_scalar

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


def _project_panel(u, v, z, basis, bounds):
    def residual(f):
        h = _one(basis(u, f))
        return np.mean((h @ _fit(h, v) - v)**2)
    f = minimize_scalar(residual, bounds=bounds, method='bounded', options={'xatol': 1e-9}).x
    return _one(basis(z, f)) @ _fit(_one(basis(u, f)), v)


def _design(study, x):
    z = _inverse(np.asarray(x, dtype=float), _calibration(study['panel_a']))
    p = study['panel_c']
    u = p['u']
    c = _project_panel(u, p['v1'], z, lambda x, f: np.c_[x, np.sin(f*x)], (1.0, 3.7))
    d = _project_panel(u, p['v2'], z, lambda x, f: np.cos(f*x), (1.5, 3.2))
    return np.c_[z, _edge(z), c, d, (c[:, :, None]*d[:, None, :]).reshape(len(z), -1)]


def _predict(h, ht, targets):
    center = h.mean(axis=0)
    scale = np.maximum(h.std(axis=0), 1e-8)
    h = (h-center)/scale
    mean = targets.mean(axis=0)
    w = np.linalg.solve(h.T @ h + 0.02*np.eye(h.shape[1]), h.T @ (targets-mean))
    return (ht-center)/scale @ w + mean


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    sparse = inputs['node_00']
    train = study['train']
    x = train['x']
    h, ht = np.split(_design(study, np.vstack((x, study['test']['x']))), [len(x)])
    traj = np.asarray(train['traj'])
    q = np.linalg.svd(traj, full_matrices=False)[2][:2].T
    pred = _predict(h, ht, np.column_stack((train['y'], traj @ q)))
    state = pred[:, 1:] @ q.T
    return {'panel_pred': np.column_stack((2*pred[:, 0]-sparse['y_hat'],
                                         (state-0.35*sparse['state_pred'])/0.65))}
