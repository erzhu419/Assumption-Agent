"""Calibrate measured coordinates, then fit shared-frequency prediction heads."""
import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(f, y):
    return np.linalg.lstsq(f, y, rcond=1e-10)[0]


def _map_features(u, w):
    return np.column_stack((_affine(u), np.sin(u*w)))


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    f = _affine(u)
    b = _fit(f, v)
    for _ in range(8):
        err = np.linalg.norm(f@b-v, axis=1)
        keep = np.argsort(err)[:max(10, int(.82*len(u)))]
        b = _fit(f[keep], v[keep])
    err = np.linalg.norm(f@b-v, axis=1)
    keep = err < max(.45, 3*np.median(err))

    def optimize(sel, starts):
        us, vs = u[sel], v[sel]
        def residual(w):
            g = _map_features(us, w)
            return (g@_fit(g, vs)-vs).ravel()
        best = min((least_squares(residual, start, bounds=(.6, 6.), max_nfev=100)
                    for start in starts), key=lambda result: result.cost)
        return best.x, _fit(_map_features(us, best.x), vs)

    w, b = optimize(keep, [np.full(u.shape[1], t) for t in (1.7, 2.5, 3.4)])
    err = np.linalg.norm(_map_features(u, w)@b-v, axis=1)
    good = err < max(.03, 4*np.median(err))
    if np.sum(good) >= 2*u.shape[1]+4:
        w, b = optimize(good, [w])
    return w, b


def _invert(x, warm, w, b):
    d = x.shape[1]
    lin, sin = b[1:d+1], b[d+1:]
    q = .75*np.linalg.solve(lin.T, (x-b[0]).T).T + .25*warm
    for _ in range(24):
        r = q@lin + np.sin(q*w)@sin + b[0] - x
        if np.max(np.abs(r), initial=0.) < 1e-10:
            break
        jac = (lin + (w*np.cos(q*w))[:, :, None]*sin).transpose(0, 2, 1)
        delta = np.linalg.solve(jac + 1e-10*np.eye(d), r[:, :, None])[..., 0]
        step = np.minimum(1., .4/np.maximum(1e-12, np.max(abs(delta), axis=1)))
        for _ in range(8):
            cand = q - step[:, None]*delta
            r1 = cand@lin + np.sin(cand*w)@sin + b[0] - x
            bad = np.sum(r1*r1, axis=1) > np.sum(r*r, axis=1)
            if not np.any(bad):
                break
            step[bad] *= .5
        q = np.where(bad[:, None], q, cand)
    return q


def _head_features(q, w):
    return np.column_stack((_affine(q), np.sin(q*w), np.cos(q*w)))


def _heads(q, query, y, traj):
    _, s, vt = np.linalg.svd(traj, full_matrices=False)
    basis = vt[:2]
    scores = traj@basis.T
    scale = np.maximum(s[:2]/np.sqrt(len(q)), .025)
    target = np.column_stack((y, scores/scale))
    def residual(w):
        f = _head_features(q, w)
        return (f@_fit(f, target)-target).ravel()
    best = min((least_squares(residual, np.full(q.shape[1], t), bounds=(1., 5.), max_nfev=90)
                for t in (2.5, 3., 3.6)), key=lambda result: result.cost)
    pred = _head_features(query, best.x)@_fit(_head_features(q, best.x), target)
    return pred[:, 0], (pred[:, 1:]*scale)@basis


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    hybrid = inputs['node_00']['hybrid']
    train, test = study['train'], study['test']
    x = np.vstack((train['x'], test['x']))
    d = x.shape[1]
    block = (d+1)*(d+2)*(d+3)//6
    warm = hybrid[:, block+1:block+1+d]
    w, b = _calibrate(study['panel_a'])
    q = _invert(x, warm, w, b)
    n = len(train['x'])
    y, z = _heads(q[:n], q[n:], train['y'], train['traj'])
    end = z[:, -1]
    return {'y_hat': y, 'state_pred': z,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
