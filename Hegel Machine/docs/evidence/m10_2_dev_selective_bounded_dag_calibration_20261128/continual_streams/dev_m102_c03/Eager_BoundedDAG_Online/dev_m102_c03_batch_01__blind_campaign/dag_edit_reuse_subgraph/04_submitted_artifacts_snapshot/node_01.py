"""Calibrated residual readout for the fixed two-branch combination."""
from itertools import combinations
import numpy as np
from scipy.interpolate import RBFInterpolator


def _calibrate(panel, x):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    keep = np.ones(len(u), dtype=bool)
    for _ in range(5):
        coeff = np.array([np.linalg.lstsq(np.c_[np.ones(keep.sum()), u[keep, j]], v[keep, j], rcond=None)[0] for j in range(u.shape[1])])
        r = np.linalg.norm(v - coeff[:, 0] - u * coeff[:, 1], axis=1)
        new = r <= max(0.002, 3 * np.median(r))
        if new.sum() < max(5, len(u)//2) or np.array_equal(new, keep):
            break
        keep = new
    return (x - coeff[:, 0]) / coeff[:, 1]


def _blend_design(z):
    s = [np.ones(len(z)), z]
    for u in z.T:
        s.extend([u**2, u**3, u**4, np.tanh(u), np.arctan(u), u/(1+abs(u)), np.exp(-u*u/2)])
    groups = []
    for i, j in combinations(range(4), 2):
        u, v = z[:, i], z[:, j]
        g = [u*v, u*u*v, u*v*v, u*u*v*v, u**3*v, u*v**3, u*v/(1+.5*(u*u+v*v))]
        for sign in (-1, 1):
            w = (u+sign*v)/1.4
            g.extend([np.sin(w), np.cos(w), np.tanh(w), w/(1+abs(w)), np.exp(-w*w/2)])
        groups.append(np.column_stack(g))
    f = np.column_stack(s+groups+[np.prod(z[:, j], axis=1) for j in combinations(range(4), 3)])
    slow = np.column_stack(s+[groups[j] for j in (0, 1, 3)])
    return f, slow


def _residual_readout(x, xt, y, t, panel, times):
    n = len(x)
    both = np.vstack([x, xt])
    z = _calibrate(panel, both)
    def warp(v, strength):
        return v + strength * np.expm1(-.2962*v*v)

    q = -np.expm1(-times[:, None] * np.array([.392, 1.857]))
    amps = np.linalg.lstsq(q, t.T, rcond=None)[0].T
    target = np.column_stack([y, amps])
    configs = [(warp(both,-.25),'inverse_quadratic',3,.25,.0001),
               (both,'quintic',2,1.,.005),
               (warp(z,-.25),'multiquadric',3,.5,.0001),
               (warp(z,-.50),'gaussian',3,.5,.0001)]
    pred = []
    for v, kernel, degree, eps, reg in configs:
        degree = min(degree, 2 if n < 35 else degree)
        pred.append(RBFInterpolator(v[:n], target, kernel=kernel, degree=degree, epsilon=eps, smoothing=reg)(v[n:]))
    d = pred[0]*np.array([.35,.25,.25]) + pred[1]*np.array([.40,.10,.10]) + pred[2]*np.array([.25,.25,.25]) + pred[3]*np.array([0.,.40,.40])

    # Fit the complementary contribution with the fixed branch as an offset.
    # Evaluating the offset design locally needs only the declared raw input.
    f, slow = _blend_design((both-x.mean(0))/np.maximum(x.std(0),1e-12))
    scalar_ids = (2,5,10,14,16,23,24,26,33,35,40,50,51,52,54,61,63,64,65,67,77,78,85,87,92,93,96,101,102,104,106,111,119,128,132,137)
    fast_ids = (7,9,10,24,30,31,35,36,39,40,51,54,61,68,71,72,85,86,96,101,104,106,111,112,123,127,128,134)
    for col, ids, ratio in ((0, scalar_ids, 1.), (2, fast_ids, .35/.65)):
        a = f[:, (0,)+tuple(j+1 for j in ids)]
        coeff = np.linalg.lstsq(a[:n], target[:, col], rcond=None)[0]
        d[:,col] = (1+ratio)*d[:,col] - ratio*(a[n:]@coeff)
    slow = slow / np.maximum(np.linalg.norm(slow[:n], axis=0), 1e-8)
    dual = np.linalg.solve(slow[:n]@slow[:n].T+.02*np.eye(n), amps[:,0])
    d[:,1] = d[:,1]/.65 - (.35/.65)*(slow[n:]@slow[:n].T@dual)
    return np.column_stack([d[:,0], d[:,1:]@q.T])


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    tr = study['train']
    x, xt = np.asarray(tr['x']), np.asarray(study['test']['x'])
    panel_pred = _residual_readout(x, xt, np.asarray(tr['y']), np.asarray(tr['traj']), study['panel_a'], np.asarray(study['eval_times']))
    return {'panel_pred': panel_pred}
