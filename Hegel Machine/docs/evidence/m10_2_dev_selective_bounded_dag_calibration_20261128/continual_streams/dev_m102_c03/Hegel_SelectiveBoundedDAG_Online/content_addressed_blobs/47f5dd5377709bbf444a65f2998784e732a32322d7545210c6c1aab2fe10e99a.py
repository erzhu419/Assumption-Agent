import numpy as np
from itertools import combinations
from scipy.optimize import least_squares


def _one(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(h, y):
    return np.linalg.lstsq(h, y, rcond=None)[0]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    w = np.ones(len(u))
    for _ in range(20):
        b = np.stack([_fit(_one(u[:, j:j+1])*w[:, None], v[:, j]*w) for j in range(4)])
        r = np.linalg.norm(v - b[:, 0] - u*b[:, 1], axis=1)
        w = 1./np.hypot(1., r/max(np.median(r), .0002))
    return (x-b[:, 0])/b[:, 1]


def _branch(u, v, z, kind):
    def design(x, p):
        a = x*p[:4]+p[4:]
        nonlin = np.tanh(a) if kind == 1 else np.expm1(-a*a)
        return _one(np.column_stack((x, nonlin)))

    def residual(p):
        h = design(u, p)
        return (h@_fit(h, v)-v).ravel()

    p = least_squares(residual, np.r_[np.full(4, 1.49 if kind == 1 else .542), np.zeros(4)],
                      bounds=(np.r_[np.full(4, .15), np.full(4, -.1)], np.r_[np.full(4, 3.), np.full(4, .1)]),
                      max_nfev=60).x
    return design(z, p) @ _fit(design(u, p), v)


def _predict(study):
    tr = study['train']
    n = len(tr['x'])
    x = np.vstack((tr['x'], study['test']['x']))
    z = _calibrate(study['panel_a'], x)
    p = study['panel_c']
    c = _branch(p['u'], p['v1'], z, 1)
    d = _branch(p['u'], p['v2'], z, 2)
    h0 = _one(np.column_stack((z, c, d, np.sum(c*d, axis=1))))
    h1 = _one(np.column_stack((z, c, d, *[a[:, i]*a[:, j] for a in (c,d) for i in range(4) for j in range(i, 4)],
                               (c[:, :, None]*d[:, None, :]).reshape(len(z), 16))))
    t = np.asarray(study['eval_times'])
    q = -np.expm1(-t[:, None]*np.array([.392, 1.857]))
    modes = np.linalg.qr(q, mode='reduced')[0]
    y = np.column_stack((tr['y'], tr['traj']@modes))

    def ridge(h, reg):
        a = h / np.maximum(np.linalg.norm(h[:n], axis=0), 1e-10)
        k = a[:n]@a[:n].T
        return a[n:]@a[:n].T@np.linalg.solve(k+reg*np.eye(n), y)

    a, b = ridge(h0, .00003), ridge(h1, .0003)
    pred = b.copy()
    pred[:, 0] = .65*a[:, 0]+.35*b[:, 0]
    return np.column_stack((pred[:, 0], pred[:, 1:]@modes.T))


def _sparse_features(z):
    s = [np.ones(len(z)), z]
    for j in range(4):
        u = z[:, j:j+1]
        s.extend([u**2, u**3, u**4, np.tanh(u), np.arctan(u), u/(1+abs(u)), np.exp(-u*u/2)])
    groups = []
    for i, j in combinations(range(4), 2):
        u, v = z[:, i:i+1], z[:, j:j+1]
        g = [u*v, u*u*v, u*v*v, u*u*v*v, u**3*v, u*v**3, u*v/(1+.5*(u*u+v*v))]
        for sign in (-1, 1):
            w = (u+sign*v)/1.4
            g.extend([np.sin(w), np.cos(w), np.tanh(w), w/(1+abs(w)), np.exp(-w*w/2)])
        groups.append(np.column_stack(g))
    return s, groups


def _sparse_baseline(study):
    tr = study['train']
    x = np.asarray(tr['x'])
    xt = np.asarray(study['test']['x'])
    n = len(x)
    z = (np.vstack([x, xt])-x.mean(0))/np.maximum(x.std(0), 1e-12)
    q = -np.expm1(-np.asarray(study['eval_times'])[:, None]*np.array([.392, 1.857]))
    a = np.linalg.lstsq(q, tr['traj'].T, rcond=None)[0].T
    s, g = _sparse_features(z)
    f = np.column_stack(s+g+[np.prod(z[:, j], axis=1) for j in combinations(range(4), 3)])[:, 1:]
    ys = (2,5,10,14,16,23,24,26,33,35,40,50,51,52,54,61,63,64,65,67,77,78,85,87,92,93,96,101,102,104,106,111,119,128,132,137)
    fs = (7,9,10,24,30,31,35,36,39,40,51,54,61,68,71,72,85,86,96,101,104,106,111,112,123,127,128,134)

    def ols(ids, y):
        mat = np.column_stack([np.ones(len(z)), f[:, ids]])
        return mat[n:] @ np.linalg.lstsq(mat[:n], y, rcond=None)[0]

    m = np.column_stack(s+[g[j] for j in (0, 1, 3)])
    m = m/np.maximum(np.linalg.norm(m[:n], axis=0), 1e-8)
    slow = m[n:] @ m[:n].T @ np.linalg.solve(m[:n] @ m[:n].T+.02*np.eye(n), a[:, 0])
    y = ols(ys, tr['y'])
    pred = np.column_stack([slow, ols(fs, a[:, 1])]) @ q.T
    return {'y_hat': y, 'state_pred': pred}


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    fitted = _predict(study)
    baseline = _sparse_baseline(study)
    base = np.column_stack((baseline['y_hat'], baseline['state_pred']))
    weight = np.r_[.5, np.repeat(.65, fitted.shape[1]-1)]
    return {'panel_pred': base + (fitted-base)/weight}
