"""Recover the calibrated operator composition and fit separate readouts."""
import numpy as np
from scipy.optimize import least_squares


def _intercept(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _calibrate(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    j = np.arange(u.shape[1])
    k = np.roll(j, -1)
    # Each channel has a cubic self term and a cyclic adjacent-channel coupling.
    def features(a):
        return np.stack((np.ones_like(a), a, a[:, k], a * a[:, k], a**3), axis=-1)

    d = features(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(10):
        c = np.column_stack([_ls(d[keep, i], v[keep, i]) for i in j])
        e = np.linalg.norm(np.einsum('nij,ji->ni', d, c) - v, axis=1)
        new = e <= max(.001, 3 * np.median(e))
        if new.sum() < 8 or np.array_equal(new, keep):
            break
        keep = new
    linear = np.diag(c[1]); linear[j, k] = c[2]
    z = np.linalg.solve(linear, (x - c[0]).T).T
    for n, r in enumerate(x):
        def residual(a):
            return np.einsum('ij,ji->i', features(a[None])[0], c) - r
        z[n] = least_squares(residual, z[n], gtol=1e-12).x
    return z


def _operator(panel, w):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v1'))
    def design(a, q):
        return _intercept(np.column_stack((a, np.arcsinh(a * np.exp(q[0])))))
    def residual(q):
        a = design(u, q)
        return (a @ _ls(a, v) - v).ravel()
    q = least_squares(residual, [1.2], bounds=(-2, 1.5)).x
    return design(w, q) @ _ls(design(u, q), v)


def _features(study, x):
    z = _calibrate(study['panel_a'], x)
    panel = study['panel_b']
    w = _intercept(z) @ _ls(_intercept(panel['u']), panel['v'])
    b = _operator(study['panel_c'], w)
    base = _intercept(np.column_stack((z, b)))
    cross = z[:, :, None] * b[:, None, :]
    return np.column_stack((base, np.trace(cross, axis1=1, axis2=2))), np.column_stack((base, cross.reshape(len(z), -1)))


def solve_batch(study):
    train = study['train']
    n = len(train['x'])
    fy, fs = _features(study, np.concatenate((train['x'], study['test']['x']), axis=0))
    y_hat = fy[n:] @ _ls(fy[:n], train['y'])
    trajectory = np.asarray(train['traj'], dtype=float)
    modes = np.linalg.svd(trajectory, full_matrices=False)[2][:2]
    state_pred = (fs[n:] @ _ls(fs[:n], trajectory @ modes.T)) @ modes
    last = state_pred[:, -1]
    return dict(y_hat=y_hat, state_pred=state_pred,
                diagnostics_vec=np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()]))
