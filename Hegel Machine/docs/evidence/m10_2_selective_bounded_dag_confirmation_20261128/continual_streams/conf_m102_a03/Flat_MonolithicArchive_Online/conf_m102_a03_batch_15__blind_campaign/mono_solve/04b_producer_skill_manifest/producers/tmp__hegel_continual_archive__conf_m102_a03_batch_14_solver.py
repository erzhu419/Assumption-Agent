"""Robust inverse calibration, panel operators, and separate scalar and trajectory readouts."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    a = u.ravel()
    b = np.roll(u, 2, axis=1).ravel()
    channel = np.tile(np.eye(4), (len(u), 1))

    def design(p):
        return np.column_stack((channel, channel * a[:, None],
                                np.tanh(p[0] * a), np.sin(p[1] * b)))

    discrepancy = np.linalg.norm(v - u, axis=1)
    mask = discrepancy <= 4 * max(np.median(discrepancy), 1e-8)
    if mask.sum() < 8:
        mask[:] = True
    p = np.array([1.8, 2.4])
    for _ in range(5):
        keep = np.repeat(mask, 4)
        target = v.ravel()[keep]

        def residual(q):
            f = design(q)[keep]
            return f @ _fit(f, target) - target

        p = least_squares(residual, p, bounds=([0.5, 1.5], [3.5, 3.5]),
                          ftol=1e-12, gtol=1e-12, xtol=1e-12).x
        f = design(p)
        c = _fit(f[keep], target)
        err = np.linalg.norm((f @ c).reshape(v.shape) - v, axis=1)
        updated = err <= max(6 * np.median(err), 1e-7)
        if updated.sum() < 8 or np.array_equal(updated, mask):
            break
        mask = updated
    return p, c


def _decode(x, cal):
    p, c = cal
    bias, gain = c[:4], c[4:8]
    h, k = c[8:]
    i = np.arange(4)
    j = (i + 2) % 4

    def residual(z, x):
        return bias + gain * z + h * np.tanh(p[0] * z) + k * np.sin(p[1] * z[j]) - x

    def jac(z, x):
        q = np.tanh(p[0] * z)
        g = np.diag(gain + h * p[0] * (1 - q*q))
        g[i, j] = k * p[1] * np.cos(p[1] * z[j])
        return g

    z = [least_squares(residual, (row - bias) / gain, jac=jac, args=(row,),
                       gtol=1e-12, ftol=1e-12, xtol=1e-12).x for row in x]
    return np.asarray(z).reshape(-1, 4)



def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _panel(panel, key, family):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', key))
    if family == 'threshold':
        def design(x, p):
            return _affine(np.sign(x) * np.maximum(abs(x)-p[0],0.)**p[1])
        start, bounds = [.35, 1.5], ([0.,.3],[.9,3.])
    else:
        def design(x, p):
            return _affine(np.column_stack((x, np.cos(p[0]*x))))
        start, bounds = [1.3], ([.2],[3.])
    def residual(p):
        a = design(u,p)
        return (a @ _fit(a,v) - v).ravel()
    p = least_squares(residual,start,bounds=bounds,ftol=1e-12,xtol=1e-12,gtol=1e-12).x
    coef = _fit(design(u,p),v)
    return lambda x: design(x,p) @ coef


def _features(study,x):
    z = _decode(x,_calibration(study['panel_a']))
    b = _panel(study['panel_b'],'v','threshold')(z)
    d = _panel(study['panel_c'],'v2','cos')(z)
    q = z*np.roll(b,1,axis=1)
    h = _affine(np.c_[b,q.sum(axis=1)])
    return h,z,b,d,q


def _scalar_predict(h, d, q, y):
    # Penalize the auxiliary interaction, not the calibrated main effects.
    n = len(y)
    interaction = (d[:, :, None] * q[:, None, :]).reshape(len(h), 16)
    a = np.column_stack((h, interaction))
    scale = np.sqrt(np.mean(a[:n]**2, axis=0)).clip(1e-9)
    a = a / scale
    penalty = np.diag(np.r_[np.zeros(h.shape[1]), np.ones(16)])
    coef = _fit(np.vstack((a[:n], penalty)), np.r_[y, np.zeros(a.shape[1])])
    return a[n:] @ coef


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    h,z,b,d,q = _features(study,np.concatenate((train['x'],study['test']['x']),axis=0))
    y_hat = _scalar_predict(h,d,q,train['y'])
    state_pred = h[n:] @ _fit(h[:n],train['traj'])
    diagnostics_vec = np.array([y_hat.mean(),y_hat.std(),state_pred[:,-1].mean(),state_pred[:,-1].std()])
    return dict(y_hat=y_hat,state_pred=state_pred,diagnostics_vec=diagnostics_vec)
