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


def _quadratic(x):
    i, j = np.triu_indices(x.shape[1])
    return np.column_stack((_aff(x), x[:, i]*x[:, j]))


def _response(x, y, q):
    """Profile a quadratic readout plus two saturating projections."""
    def design(z, p):
        h = z[:, 2] + z[:, [0, 1, 3]] @ p[:3] + p[3]
        return np.column_stack((_quadratic(z), np.tanh(p[4]*h),
                                np.tanh(z @ p[5:9] + p[9])))

    penalty = np.diag(np.r_[np.zeros(5), np.full(10, .003), [.008, .008]])
    target = np.r_[y, np.zeros(len(penalty))]

    def profile(p):
        d = np.vstack((design(x, p), penalty))
        w = _ls(d, target)
        return d @ w - target, w

    starts = [a+b for a in [[-.52,-.15,-.12,0,1.8], [-.5,-.15,-.12,-.5,2.15]]
              for b in [[1,0,-1,0,0], [0,1,0,-1,0], [-.5,-.15,1,-.12,0]]]
    bounds = ([-3]*4 + [.2] + [-5]*5, [3]*4 + [6.] + [5]*5)
    fit = min((least_squares(lambda p: profile(p)[0], p, bounds=bounds,
                            max_nfev=100) for p in starts), key=lambda r: r.cost)
    return design(q, fit.x) @ profile(fit.x)[1]


def solve_batch(study: dict) -> dict:
    train = study['train']
    c = _calibrate(study['panel_a'])
    x = _inverse(train['x'], c)
    q = _inverse(study['test']['x'], c)
    y = _response(x, np.asarray(train['y']), q)
    state = _aff(q) @ _ls(_aff(x), np.asarray(train['traj']))
    last = state[:, -1]
    diagnostics = np.array([y.mean(), y.std(), last.mean(), last.std()]) if len(y) else np.zeros(4)
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diagnostics}
