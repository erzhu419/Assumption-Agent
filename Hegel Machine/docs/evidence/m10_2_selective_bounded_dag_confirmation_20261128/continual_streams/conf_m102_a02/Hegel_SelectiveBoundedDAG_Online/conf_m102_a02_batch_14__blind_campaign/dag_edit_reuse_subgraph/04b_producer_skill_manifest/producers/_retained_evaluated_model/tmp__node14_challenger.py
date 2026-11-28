"""Invert the reference calibration and fit separate scalar and trajectory readouts."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar, root


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _calibrate(panel, obs, upstream):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    gain, offset = upstream['gain'], upstream['offset']
    delta = np.max(np.abs(v - u * gain - offset), axis=1)
    m = np.median(delta)
    good = delta < m + max(.42, 5 * np.median(abs(delta - m)))
    def design(q, p, j):
        a, b = q[:, j], q[:, (j + 2) % 4]
        return np.c_[np.ones(len(q)), a, b, np.sin(p[0]*a), np.sin(p[1]*b), a**3]
    def fit(p, mask):
        ds = [design(u[mask], p, j) for j in range(4)]
        co = [_ls(a, v[mask, j]) for j, a in enumerate(ds)]
        return co, np.column_stack([a @ w - v[mask, j] for j, (a,w) in enumerate(zip(ds,co))])
    def opt(mask):
        return least_squares(lambda p: fit(p, mask)[1].ravel(), [2.4, 2.4], bounds=([1., 1.], [4., 4.])).x
    p = opt(good)
    c, _ = fit(p, good)
    def forward(q):
        return np.column_stack([design(q, p, j) @ c[j] for j in range(4)])
    err = np.max(abs(forward(u) - v), axis=1)
    med = np.median(err)
    good = err < max(.015, 6 * med)
    if good.sum() >= 20:
        p = opt(good)
        c, _ = fit(p, good)
    def inv(row):
        init = (row - offset) / gain
        fun = lambda x: forward(x[None,:])[0] - row
        sol = root(fun, init)
        if np.linalg.norm(fun(sol.x)) > 1e-7:
            sol = least_squares(fun, np.clip(init, -1.49, 1.49), bounds=(-1.5, 1.5))
        return sol.x
    return np.array([inv(row) for row in obs])


def _mapped(q, fitted):
    scale, power, exponent = fitted['p']
    logabs = np.log(np.maximum(np.abs(q), np.finfo(float).tiny))
    h = q * np.exp(exponent * np.logaddexp(0, power * (np.log(scale) + logabs)))
    return np.c_[np.ones(len(q)), q, h] @ fitted['w']


def _scalar(q, b, c, y, k):
    n = len(y)
    one = np.ones(len(q))
    def designs():
        for a in (q, b):
            scores = list(a.T)
            scores += [a[:,i] + s*a[:,j] for i in range(4) for j in range(i+1,4) for s in (-1,1)]
            for h in scores:
                for strength in (k, 1.25):
                    g = 1/(1 + strength*h*h)
                    bg = b*g[:,None]
                    yield np.c_[one, bg, c]
                    yield np.c_[one, bg, c, bg*h[:,None]**3]
                    yield np.c_[one, bg, c, h, h*g, h*h*g]
                    yield np.c_[one, bg, c, b]
                    yield np.c_[one, bg, c, h**3*g]
                    yield np.c_[one, bg, c, h**3]
                    yield np.c_[one, bg, c, g]
                    yield np.c_[one, bg, c*g[:,None]]
        yield np.c_[one,b,c]
    results = []
    for a in designs():
        x = a[:n]
        u,s,vt = np.linalg.svd(x, full_matrices=False)
        keep = s > s[0]*1e-9
        u = u[:,keep]
        d = u.T@y
        e = abs(y-u@d)/np.maximum(1-np.sum(u*u,axis=1),.05)
        score = e.mean()
        results.append((score, a[n:] @ (vt[keep].T @ (d/s[keep]))))
    results.sort(key=lambda t:t[0])
    best = results[0][0]
    sel = [(s,p) for s,p in results[:8] if s < best + .006 or s == best]
    w = np.exp(-(np.array([s for s,p in sel])-best)/.0015)
    return np.average([p for s,p in sel], axis=0, weights=w)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    fitted = inputs['node_00']
    train = study['train']
    n = len(train['x'])
    q = _calibrate(study['panel_a'], np.vstack((train['x'], study['test']['x'])), fitted)
    b = _mapped(q, fitted)
    c = b*np.roll(q, -1, axis=1)
    y = _scalar(q, b, c, np.asarray(train['y']), float(study['telemetry_vec'][2]))
    design = np.c_[np.ones(len(q)), b, c]
    state = design[n:] @ _ls(design[:n], train['traj'])
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
