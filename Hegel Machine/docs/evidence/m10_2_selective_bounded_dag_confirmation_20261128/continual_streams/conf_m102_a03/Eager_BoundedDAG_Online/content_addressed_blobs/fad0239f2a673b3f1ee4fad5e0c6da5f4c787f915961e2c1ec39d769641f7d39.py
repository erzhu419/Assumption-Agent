"""Calibrated scalar readout and reduced-rank trajectory prediction."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _readout(z):
    return np.column_stack((_affine(z), z[:, 0]*z[:, 2] + z[:, 1]*z[:, 3]))


def _scalar(z, zt, u, ut, y):
    l, lt = _readout(z), _readout(zt)
    x, xt = _affine(u), _affine(ut)
    q = np.linalg.qr(l, mode='reduced')[0]
    residual = y - q @ (q.T @ y)

    def loss(h):
        f = np.tanh(x @ h)
        f -= q @ (q.T @ f)
        d = f @ f
        if d < 1e-22:
            return residual
        return residual - f * ((f @ residual) / d)

    starts = [np.r_[0., k*np.eye(u.shape[1])[j]] for j in range(u.shape[1]) for k in (0.85, 1.8)]
    best = min((least_squares(loss, h, bounds=(-5., 5.), max_nfev=110) for h in starts), key=lambda opt: opt.cost)
    d = np.column_stack((l, np.tanh(x @ best.x)))
    dt = np.column_stack((lt, np.tanh(xt @ best.x)))
    return dt @ _fit(d, y)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    encoded = inputs['node_00']
    train = study['train']
    z, zt = (np.asarray(encoded[k], dtype=float) for k in ('z', 'zt'))
    panel = study['panel_b']
    b = _fit(_affine(np.asarray(panel['u'], dtype=float)), np.asarray(panel['v'], dtype=float))
    u, ut = (np.linalg.solve(b[1:].T, (v-b[0]).T).T for v in (z, zt))
    yh = _scalar(z, zt, u, ut, np.asarray(train['y'], dtype=float))
    q, r = np.linalg.qr(z, mode='reduced')
    left, sv, right = np.linalg.svd(q.T @ train['traj'], full_matrices=False)
    sp = zt @ _fit(r, (left[:, :2] * sv[:2]) @ right[:2])
    tail = sp[:, -1]
    return {'y_hat': yh, 'state_pred': sp,
            'diagnostics_vec': np.array([yh.mean(), yh.std(), tail.mean(), tail.std()])}
