"""Calibrated dead-zone features for scalar and trajectory prediction."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    n = len(u)
    a = u.ravel()
    b = np.roll(u, -1, axis=1).ravel()
    channel = np.tile(np.eye(4), (n, 1))
    design = np.column_stack((channel, channel * a[:, None], b, a*b, a**3))
    discrepancy = np.linalg.norm(v-u, axis=1)
    good = discrepancy < 4 * max(np.median(discrepancy), 1e-8)
    if good.sum() < 8:
        good[:] = True
    for _ in range(5):
        keep = np.repeat(good, 4)
        coef = _fit(design[keep], v.ravel()[keep])
        residual = np.linalg.norm((design @ coef).reshape(n, 4) - v, axis=1)
        refined = residual <= max(5 * np.median(residual), 1e-7)
        if refined.sum() < 8 or np.array_equal(good, refined):
            break
        good = refined
    return coef


def _decode(x, coef):
    offset, gain = coef[:4], coef[4:8]
    coupling, interaction, cubic = coef[8:]
    i = np.arange(4)
    j = (i + 1) % 4

    def residual(z, row):
        nxt = z[j]
        return offset + gain*z + coupling*nxt + interaction*z*nxt + cubic*z**3 - row

    def jacobian(z, row):
        g = np.diag(gain + interaction*z[j] + 3*cubic*z*z)
        g[i, j] = coupling + interaction*z
        return g

    z = []
    for row in x:
        result = least_squares(residual, (row-offset)/gain, jac=jacobian,
                               args=(row,), gtol=1e-12, max_nfev=80)
        if np.linalg.norm(result.fun, np.inf) > 1e-8:
            other = least_squares(residual, np.zeros(4), jac=jacobian,
                                  args=(row,), gtol=1e-12, max_nfev=100)
            if np.linalg.norm(other.fun) < np.linalg.norm(result.fun):
                result = other
        z.append(result.x)
    return np.asarray(z).reshape(-1, 4)


def _warp(x, p):
    return np.sign(x) * np.maximum(np.abs(x) - p[0], 0.0) ** p[1]


def _operator(panel):
    u, v = [np.asarray(panel[k], dtype=float) for k in ('u', 'v')]

    def residual(p):
        a = _affine(_warp(u, p))
        return (a @ _fit(a, v) - v).ravel()

    scale = np.max(np.abs(u))
    p = least_squares(residual, [0.25 * scale, 1.5],
                      bounds=([0.0, 0.5], [0.8 * scale, 3.5]),
                      ftol=1e-11, xtol=1e-11, gtol=1e-11).x
    return p, _fit(_affine(_warp(u, p)), v)


from itertools import combinations_with_replacement


def _mixed(z, h):
    return np.column_stack((_affine(h), z, *[h[:,i]*h[:,j] for i,j in combinations_with_replacement(range(4),2)], (h[:,:,None]*z[:,None,:]).reshape(-1,16)))


def _ridge(q, qt, target, penalty):
    scale = np.maximum(np.sqrt(np.mean(q*q,axis=0)),1e-7)
    a = q/scale
    return (qt/scale) @ np.linalg.solve(a.T@a+penalty*np.eye(a.shape[1]),a.T@target)


def _quadratic(z):
    return np.column_stack([z[:, i]*z[:, j]
                            for i, j in combinations_with_replacement(range(4), 2)])


def _adaptive_scalar(q, qt, y):
    scale = np.sqrt(np.mean(q*q, axis=0)).clip(1e-8)
    design = q/scale
    gram = design.T @ design
    rhs = design.T @ y
    coef = np.linalg.solve(gram + .03*np.eye(gram.shape[0]), rhs)
    for _ in range(5):
        penalty = .003/(np.abs(coef)+.035)**2
        penalty[:9] *= .1
        coef = np.linalg.solve(gram + np.diag(penalty), rhs)
    return (qt/scale) @ coef


def solve_batch(study: dict) -> dict:
    train=study['train']
    cal=_calibration(study['panel_a'])
    z=_decode(np.asarray(train['x'],dtype=float),cal)
    zt=_decode(np.asarray(study['test']['x'],dtype=float),cal)
    p,_=_operator(study['panel_b'])
    y=np.asarray(train['y'],dtype=float)
    def profile(q):
        a=_mixed(z,_warp(z,q))
        return a@_fit(a,y)-y
    py=min((least_squares(profile,start,bounds=([0,.5],[.75,4]),max_nfev=100) for start in ([.22,1.25],[.35,1.5],[.5,2])),key=lambda r:r.cost).x
    adaptive = _adaptive_scalar(_mixed(z,_warp(z,py)), _mixed(zt,_warp(zt,py)), y)
    fixed, fixed_test = [np.column_stack((_mixed(v,_warp(v,[.42,1.25])),_quadratic(v)))
                         for v in (z,zt)]
    y_hat = .75*adaptive + .25*_ridge(fixed,fixed_test,y,.03)
    state=_ridge(_mixed(z,_warp(z,p)),_mixed(zt,_warp(zt,p)),np.asarray(train['traj']),.015)
    last=state[:,-1]
    return {'y_hat':y_hat,'state_pred':state,'diagnostics_vec':np.array([y_hat.mean(),y_hat.std(),last.mean(),last.std()])}
