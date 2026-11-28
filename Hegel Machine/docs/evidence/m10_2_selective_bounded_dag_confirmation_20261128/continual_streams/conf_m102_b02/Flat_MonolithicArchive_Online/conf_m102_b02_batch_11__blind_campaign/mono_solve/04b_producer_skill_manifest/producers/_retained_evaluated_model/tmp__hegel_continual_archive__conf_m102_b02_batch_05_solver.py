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


def _b_operator(panel, w):
    """Fit a shared scalar deformation followed by affine mixing."""
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))

    def design(a, q):
        amplitude, frequency = q
        return _intercept(a - amplitude * np.sin(frequency * a))

    def residual(q):
        d = design(u, q)
        return (d @ _ls(d, v) - v).ravel()

    q = least_squares(residual, [.4, 2.9], bounds=(.005, 8.)).x
    return design(w, q) @ _ls(design(u, q), v)


def _features(study, x):
    z = _calibrate(study['panel_a'], x)
    w = _b_operator(study['panel_b'], z)
    panel = study['panel_c']
    def design(a):
        return _intercept(a * np.roll(a, -1, axis=1))
    q = design(w) @ _ls(design(panel['u']), panel['v1'])
    base = _intercept(np.column_stack((z, q)))
    cross = z[:, :, None] * q[:, None, :]
    return (np.column_stack((base, np.trace(cross, axis1=1, axis2=2))),
            np.column_stack((base, cross.reshape(len(z), -1))))


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    fy, fs = _features(study, np.vstack((train['x'], study['test']['x'])))
    y = fy[n:] @ _ls(fy[:n], train['y'])
    traj = np.asarray(train['traj'], dtype=float)
    mean = traj.mean(axis=0)
    modes = np.linalg.svd(traj - mean, full_matrices=False)[2][:2]
    state = (fs[n:] @ _ls(fs[:n], (traj - mean) @ modes.T)) @ modes + mean
    end = state[:, -1]
    return dict(y_hat=y, state_pred=state,
                diagnostics_vec=np.array([y.mean(), y.std(), end.mean(), end.std()]))
