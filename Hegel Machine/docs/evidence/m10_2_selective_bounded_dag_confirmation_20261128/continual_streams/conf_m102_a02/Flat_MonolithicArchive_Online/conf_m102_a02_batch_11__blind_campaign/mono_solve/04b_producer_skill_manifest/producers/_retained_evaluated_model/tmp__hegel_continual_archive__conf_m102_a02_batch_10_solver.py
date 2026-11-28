"""Calibrate latent inputs, then fit independent scalar and state readouts."""
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


def _operator(panel):
    """Profile linear mixing weights out of the componentwise panel fit."""
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))

    def design(x, p):
        return np.column_stack((_aff(x), x*np.exp(-(p[0]*x)**2),
                                x/(1+p[1]*np.abs(x))))

    def residual(p):
        f = design(u, p)
        return (f @ _ls(f, v) - v).ravel()

    fits = [least_squares(residual, [p, .65], bounds=([.3,.07],[7.,5.]))
            for p in (1.5, 3., 4.5)]
    p = min(fits, key=lambda fit: fit.cost).x
    weights = _ls(design(u, p), v)
    return lambda x: design(x, p) @ weights


def _quadratic(h):
    i, j = np.triu_indices(h.shape[1])
    return np.column_stack((_aff(h), h[:,i]*h[:,j]))


def _state(h, q, t, z):
    # Shared two-rate relaxation; the mode amplitudes are affine in panel B.
    f, g = _aff(h), _aff(q)
    def basis(p):
        return -np.expm1(-t[:,None]*p)
    def weights(b):
        return _ls(f, _ls(b, z.T).T)
    def residual(p):
        b = basis(p)
        return (f @ weights(b) @ b.T-z).ravel()
    p = least_squares(residual, [1.8,.42], bounds=([.85,.03],[6.,.84])).x
    b = basis(p)
    return g @ weights(b) @ b.T


def solve_batch(study: dict) -> dict:
    train = study['train']
    calibration = _calibrate(study['panel_a'])
    op = _operator(study['panel_b'])
    h = op(_inverse(train['x'], calibration))
    q = op(_inverse(study['test']['x'], calibration))
    y = _quadratic(q) @ _ls(_quadratic(h), train['y'])
    state = _state(h, q, np.asarray(study['eval_times']), np.asarray(train['traj']))
    last = state[:, -1]
    diagnostics = np.array([y.mean(), y.std(), last.mean(), last.std()]) if len(y) else np.zeros(4)
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diagnostics}
