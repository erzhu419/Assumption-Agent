"""Cyclic coordinate calibration and shared two-mode relaxation forecasts."""
import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _next(x):
    return np.roll(x, -1, axis=1)


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    w = []
    for a, b, y in zip(u.T, _next(u).T, v.T):
        f = np.column_stack((np.ones(len(a)), a, b, a*b, a**3))
        # Least-trimmed pilot: the coordinate law is overdetermined even
        # after discarding the largest residuals in each channel.
        p = _ls(f, y)
        k = max(f.shape[1]+1, int(.8*len(a)))
        for _ in range(8):
            keep = np.argsort(abs(f@p-y))[:k]
            p = _ls(f[keep], y[keep])
        r = abs(f@p-y)
        keep = r <= max(7*np.median(r), 1e-6)
        w.append(_ls(f[keep], y[keep]) if keep.sum() >= f.shape[1] else p)
    return np.asarray(w).T


def _inverse(x, w):
    c, a, b, p, d = w
    z = (x-c)/a
    j = np.arange(x.shape[1]); J = np.zeros((len(x), len(j), len(j)))
    for _ in range(20):
        nxt = _next(z)
        r = c+a*z+b*nxt+p*z*nxt+d*z**3-x
        if np.max(abs(r), initial=0.) < 1e-11:
            break
        J[:,j,j] = a+p*nxt+3*d*z*z
        J[:,j,(j+1)%len(j)] = b+p*z
        step = np.linalg.solve(J, r[...,None])[...,0]
        z -= np.clip(step, -.5, .5)
    return z


def _basis(t, traj):
    def basis(k):
        return -np.expm1(-t[:,None]*k)
    def err(k):
        B = basis(k)
        return (B@_ls(B, traj.T)-traj.T).ravel()
    rates = least_squares(err, [1.8,.4], bounds=([.8,.03],[8.,.79])).x
    return basis(rates)


def _design(x):
    i, j = np.triu_indices(x.shape[1])
    return np.column_stack((np.ones(len(x)), x, x[:, i]*x[:, j]))


def solve_batch(study: dict) -> dict:
    train = study['train']
    w = _calibration(study['panel_a'])
    x, q = [_inverse(np.asarray(m['x'], dtype=float), w)
            for m in (train, study['test'])]
    traj = np.asarray(train['traj'], dtype=float)
    B = _basis(np.asarray(study['eval_times'], dtype=float), traj)
    amplitudes = _ls(B, traj.T).T
    F, G = _design(x), _design(q)
    y = G @ _ls(F, np.asarray(train['y'], dtype=float))
    # The trajectory is close to affine after calibration. Shrink small
    # quadratic corrections to reduce leverage in coordinate-tail forecasts.
    reg = .5 * np.eye(F.shape[1])[1 + x.shape[1]:]
    coeff = _ls(np.vstack((F, reg)),
                np.vstack((amplitudes, np.zeros((len(reg), B.shape[1])))))
    z = (G @ coeff) @ B.T
    return {'y_hat': y, 'state_pred': z,
            'diagnostics_vec': np.array([y.mean(), y.std(), z[:, -1].mean(), z[:, -1].std()])}
