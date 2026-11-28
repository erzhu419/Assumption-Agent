"""Calibrated dual-operator predictor with a two-rate trajectory basis."""
import numpy as np
from scipy.optimize import least_squares


def _design(*x):
    return np.column_stack((np.ones(len(x[0])), *x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-10)[0]


def _calibrate(panel, observed):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    mask = np.max(np.abs(u-v), axis=1) < 3
    u, v = u[mask], v[mask]
    j = np.arange(4)
    eye = np.broadcast_to(np.eye(4), (len(u), 4, 4))
    base = (eye, eye*u[:, :, None])

    def profile(rate):
        a = np.concatenate((*base, np.tanh(rate[0]*u)[:, :, None],
                            np.sin(rate[1]*np.roll(u, 2, axis=1))[:, :, None]), axis=2).reshape(-1, 10)
        w = _fit(a, v.ravel())
        return a @ w-v.ravel(), w

    k = least_squares(lambda p: profile(p)[0], [1.8, 2.4], bounds=([.5, 1.5], [4., 3.3])).x
    _, w = profile(k)
    offset, gain, a, b = w[:4], w[4:8], w[8], w[9]
    z = (observed-offset)/gain
    # Fixed-point warm start; refine against the original equations below.
    for _ in range(55):
        update = (observed-offset-a*np.tanh(k[0]*z)-b*np.sin(k[1]*np.roll(z, 2, axis=1)))/gain
        if np.max(np.abs(update-z), initial=0.) < 1e-12:
            break
        z = update
    # Newton refinement (the self term is monotone and dominates the coupling).
    for _ in range(12):
        r = np.roll(z, 2, axis=1)
        h = np.tanh(k[0]*z)
        err = offset+gain*z+a*h+b*np.sin(k[1]*r)-observed
        if np.max(np.abs(err), initial=0.) < 1e-12:
            break
        jac = np.zeros((len(z),4,4))
        jac[:,j,j] = gain+a*k[0]*(1-h*h)
        jac[:,j,(j+2)%4] = b*k[1]*np.cos(k[1]*r)
        z -= np.linalg.solve(jac, err[...,None])[...,0]
    return z


_RING = ((0,1),(0,3),(1,2),(2,3))


def _panels(panel, z):
    from scipy.optimize import minimize_scalar
    u, vp, vq = (np.asarray(panel[k], dtype=float) for k in ('u', 'v1', 'v2'))

    def latent(x, k):
        return np.column_stack((x / np.sqrt(1 + k*x*x),
                                *(x[:, i]*x[:, j] for i,j in _RING)))

    def profile(target, design, bounds):
        def residual(k):
            a = design(u, k)
            return np.linalg.norm(a @ _fit(a, target) - target)
        k = minimize_scalar(residual, bounds=bounds, method='bounded').x
        return k, _fit(design(u, k), target)

    def q_design(x, k):
        return _design(latent(x, k))

    kq, cq = profile(vq, q_design, (.2, 1.5))

    def p_design(x, k):
        return _design(x, x/(1+k*x*x), latent(x, kq))

    kp, cp = profile(vp, p_design, (.2, 2.5))
    return p_design(z, kp) @ cp, q_design(z, kq) @ cq


def _time_basis(times, rates):
    return -np.expm1(-np.outer(times, rates))


def _temporal_fit(times, traj):
    def residual(rates):
        a = _time_basis(times, rates)
        return (a @ _fit(a, traj.T) - traj.T).ravel()

    rates = least_squares(residual, [1.8, .42],
                         bounds=([.75, .03], [6., .74])).x
    t = _time_basis(times, rates)
    return t, _fit(t, traj.T).T


def solve_batch(study: dict) -> dict:
    """Fit calibration and two response modes on the supplied panels and training rows."""
    train = study['train']
    y = np.asarray(train['y'], dtype=float)
    n = len(y)
    x = np.concatenate((train['x'], study['test']['x']), axis=0)
    z = _calibrate(study['panel_a'], x)
    p, q = _panels(study['panel_c'], z)
    interaction = (p[:, :, None] * q[:, None, :]).reshape(len(x), 16)
    a = _design(p, q, interaction)
    t, amplitude = _temporal_fit(study['eval_times'], train['traj'])
    prediction = a[n:] @ _fit(a[:n], np.column_stack((y, amplitude)))
    y_hat = prediction[:, 0]
    state = prediction[:, 1:] @ t.T
    diagnostics = np.array([y_hat.mean(), y_hat.std(), state[:, -1].mean(), state[:, -1].std()])
    return {'y_hat': y_hat, 'state_pred': state, 'diagnostics_vec': diagnostics}
