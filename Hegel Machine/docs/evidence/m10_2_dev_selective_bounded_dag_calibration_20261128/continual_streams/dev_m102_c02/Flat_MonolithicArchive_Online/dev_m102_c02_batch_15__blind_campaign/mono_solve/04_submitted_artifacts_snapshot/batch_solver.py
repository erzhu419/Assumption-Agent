"""Paired-coordinate calibration and panel-B scalar and dynamical readouts."""
import numpy as np
from scipy.optimize import least_squares


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(h, y):
    return np.linalg.lstsq(h, y, rcond=1e-10)[0]


def _calibrate(panel, rows):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    norm = np.linalg.norm(v, axis=1)
    keep = norm < max(4., 3*np.median(norm))
    u, v = u[keep], v[keep]

    def design(z, p):
        a, power, decay, frequency = p
        other = np.roll(z, 2, axis=-1)
        return np.stack((np.ones_like(z), z/(1+a*np.abs(z)**power)**decay,
                         np.sin(frequency*other)), axis=-1)

    def profile(p):
        h = design(u, p)
        c = np.array([_fit(h[:, j], v[:, j]) for j in range(4)])
        return (h*c).sum(axis=-1)-v, c

    p = least_squares(lambda p: profile(p)[0].ravel(), [.5,2,.5,2.4],
                      bounds=([.001,.3,.01,.5], [8,6,8,6]), gtol=1e-12).x
    _, c = profile(p)
    return np.array([least_squares(lambda z: (design(z,p)*c).sum(axis=-1)-q,
                                   q, gtol=1e-11).x for q in rows])


def _operator(panel, z):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))

    def design(x, f):
        return _aff(np.cos(f*x))

    def residual(p):
        h = design(u, p[0])
        return (h@_fit(h,v)-v).ravel()

    f = least_squares(residual, [2.6], bounds=([1.8], [3.5]), gtol=1e-12).x[0]
    return design(z,f)@_fit(design(u,f),v)


def _trajectory(h, q, y, t):
    # Fit rates to the part of the trajectories explained by the spatial map.
    amplitudes = _fit(h, y)
    projected = (h@amplitudes).T

    def temporal(p):
        return -np.expm1(-t[:,None]*p)

    def residual(p):
        f = temporal(p)
        return (f@_fit(f, projected)-projected).ravel()

    p = least_squares(residual, [.4,1.8], bounds=([.01,.85],[.85,10]),
                      gtol=1e-11).x
    f = temporal(p)
    return q @ _fit(f, amplitudes.T).T @ f.T


def solve_batch(study: dict) -> dict:
    tr = study['train']
    n = len(tr['x'])
    z = _calibrate(study['panel_a'], np.concatenate((tr['x'], study['test']['x'])))
    b = _operator(study['panel_b'], z)
    h = _aff(b)
    paired = b[:,0]*b[:,2] + b[:,1]*b[:,3]
    s = np.column_stack((h, paired))
    y = s[n:] @ _fit(s[:n], tr['y'])
    state = _trajectory(h[:n], h[n:], tr['traj'], np.asarray(study['eval_times']))
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
