from itertools import combinations, combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import lars_path


def _aug(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _poly(x, degrees):
    return np.column_stack([np.prod(x[:, js], axis=1) for k in degrees
                            for js in combinations_with_replacement(range(4), k)])


def _operators(study):
    a, b, c = (study['panel_' + k] for k in 'abc')
    uv = a['u'], a['v']
    ab = []
    for j in range(4):
        u, v = (z[:, j] for z in uv)
        m = np.abs(v - u) < 1.5
        ab.append(_ls(_aug(u[m, None]), v[m]))
    bias, gain = np.asarray(ab).T
    B = _ls(_aug(b['u']), b['v'])
    invB = np.linalg.inv(B[1:])
    def fit(fn, output, p0):
        u = c['u']
        def err(p):
            f = fn(u, p)
            return (f @ _ls(f, output) - output).ravel()
        p = least_squares(err, p0, max_nfev=100).x
        coeff = _ls(fn(u, p), output)
        return lambda x: fn(x, p) @ coeff
    z = fit(lambda x, p: _aug(np.c_[x/(1+np.exp(p[0])*abs(x)**np.exp(p[1])), x]), c['v1'], [0., 0.])
    w = fit(lambda x, p: _aug(np.c_[x/(1+np.exp(p[0])*x*x), x*x/(1+np.exp(p[1])*x*x)]), c['v2'], [-3., -3.])
    return bias, gain, B, invB, z, w


def _cross(a, b):
    return (a[:, :, None] * b[:, None, :]).reshape(len(a), -1)


def _dictionary(d, z, w):
    blocks = [d, z, w, _poly(d, [2]), _poly(d, [3])]
    for idx, f in enumerate([d, z, w, z+w, z-w, d+z, d+w, z*w]):
        if idx < 3:
            f = np.column_stack([f]+[f[:, i]+sg*f[:, j] for i,j in combinations(range(4), 2) for sg in [-1,1]])
        for k in [.5, 1, 1.7, 2.5]:
            blocks.extend([np.sin(k*f), np.cos(k*f)-1, np.tanh(k*f), f/(1+k*abs(f)), f/(1+k*f*f), f*f/(1+k*f*f)])
        blocks.extend([f*f, f**3, abs(f)])
        if idx < 3:
            blocks.append(np.column_stack([f[:, i]*f[:, j] for i,j in combinations(range(f.shape[1]), 2)]))
    return np.column_stack(blocks+[_cross(z,w),_cross(z,d),_cross(w,d)])


def _scaled(f, n, eps):
    return (f-f[:n].mean(axis=0))/np.maximum(f[:n].std(axis=0), eps)


def _scalar(x, y, ops):
    bias, gain, B, _, z, w = ops
    d = _aug((x-bias)/gain) @ B
    f = _dictionary(d, z(d), w(d))
    n = len(y)
    p = np.clip((f-f[:n].mean(0))/(f[:n].std(0)+1e-9), -30, 30)
    _, _, coeffs = lars_path(p[:n], y-y.mean(), method='lasso', max_iter=66)
    return p[n:] @ coeffs[:, min(65, coeffs.shape[1]-1)] + y.mean()


def _trajectory(x, t, ops):
    bias, gain, B, invB, z, w = ops
    d = (x-bias)/gain
    h = d + (w(d)-B[0]) @ invB
    n = len(t)
    f = _scaled(np.c_[_poly(h, [1,2]),_poly(d, [1,2]),_cross(d,h)], n, 1e-7)
    a, b = f[:n], f[n:]
    centered = t-t.mean(axis=0)
    vt = np.linalg.svd(centered, full_matrices=False)[2][:2]
    coeff = np.linalg.solve(a.T@a + .15*np.eye(a.shape[1]), a.T@centered@vt.T)
    return b @ coeff @ vt + t.mean(axis=0)


def solve_batch(study: dict) -> dict:
    tr = study['train']
    x = np.vstack((tr['x'], study['test']['x']))
    ops = _operators(study)
    y = _scalar(x, tr['y'], ops)
    state = _trajectory(x, tr['traj'], ops)
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
